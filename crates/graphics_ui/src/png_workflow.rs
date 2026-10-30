use editor::Editor;
use gpui::{
    App, AppContext as _, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable,
    Render, ScrollHandle, Subscription, Task, WeakEntity, Window, actions,
};
use ui::{ListItem, prelude::*};
use workspace::{DismissDecision, ModalView, Workspace};

use crate::interop::{
    GraphitePngAssetChoice, GraphitePublishedPngChoice, GraphiteSimDocumentChoice,
};

actions!(
    graphite,
    [
        /// Opens an authorized PNG as editable Graphite artwork.
        OpenPngEditor,
        /// Opens saved editable Graphite artwork from authorized project output assets.
        OpenSimDocument,
        /// Opens an immutable published PNG version as linked editable Graphite artwork.
        OpenPublishedPng
    ]
);

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _cx| {
        workspace.register_action(|workspace, _: &OpenPngEditor, window, cx| {
            toggle_chooser(ChooserKind::InputPng, workspace, window, cx);
        });
        workspace.register_action(|workspace, _: &OpenSimDocument, window, cx| {
            toggle_chooser(ChooserKind::SavedSim, workspace, window, cx);
        });
        workspace.register_action(|workspace, _: &OpenPublishedPng, window, cx| {
            toggle_chooser(ChooserKind::PublishedPng, workspace, window, cx);
        });
    })
    .detach();
}

fn toggle_chooser(
    kind: ChooserKind,
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let switching_kind = workspace
        .active_modal::<AuthorizedPngChooser>(cx)
        .is_some_and(|modal| modal.read(cx).kind != kind);
    if switching_kind && !workspace.hide_modal(window, cx) {
        return;
    }
    let workspace_handle = cx.entity().downgrade();
    workspace.toggle_modal(window, cx, move |window, cx| match kind {
        ChooserKind::InputPng => AuthorizedPngChooser::new(workspace_handle, window, cx),
        ChooserKind::SavedSim | ChooserKind::PublishedPng => {
            AuthorizedPngChooser::new_for(kind, workspace_handle, window, cx)
        }
    });
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ChooserKind {
    InputPng,
    SavedSim,
    PublishedPng,
}

impl ChooserKind {
    fn selector(self, suffix: &str) -> String {
        match self {
            Self::InputPng => format!("GRAPHITE-PNG-{suffix}"),
            Self::SavedSim => format!("GRAPHITE-SIM-{suffix}"),
            Self::PublishedPng => format!("GRAPHITE-PUBLISHED-PNG-{suffix}"),
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::InputPng => "Open PNG in Graphite",
            Self::SavedSim => "Open saved Graphite document",
            Self::PublishedPng => "Open published PNG in Graphite",
        }
    }

    fn placeholder(self) -> &'static str {
        match self {
            Self::InputPng => "Search authorized PNGs",
            Self::SavedSim => "Search saved .sim documents",
            Self::PublishedPng => "Search published PNG versions",
        }
    }
}

#[derive(Clone)]
enum AssetChoice {
    Png(GraphitePngAssetChoice),
    Sim(GraphiteSimDocumentChoice),
    PublishedPng(Box<GraphitePublishedPngChoice>),
}

impl AssetChoice {
    fn label(&self) -> String {
        match self {
            Self::Png(choice) => choice.label(),
            Self::Sim(choice) => choice.label(),
            Self::PublishedPng(choice) => choice.label(),
        }
    }

    fn byte_size(&self) -> u64 {
        match self {
            Self::Png(choice) => choice.byte_size(),
            Self::Sim(choice) => choice.byte_size(),
            Self::PublishedPng(choice) => choice.byte_size(),
        }
    }
}

struct ChoicePage {
    choices: Vec<AssetChoice>,
    next_offset: Option<usize>,
}

struct AuthorizedPngChooser {
    kind: ChooserKind,
    workspace: WeakEntity<Workspace>,
    search_editor: Entity<Editor>,
    choices: Vec<AssetChoice>,
    selected_index: Option<usize>,
    scroll_handle: ScrollHandle,
    offset: usize,
    previous_offsets: Vec<usize>,
    next_offset: Option<usize>,
    query_generation: u64,
    loading: bool,
    opening: bool,
    dismissed: bool,
    diagnostic: Option<String>,
    query_task: Option<Task<()>>,
    open_task: Option<Task<()>>,
    open_cancellation: Option<std::sync::Arc<sim_tensor::CancellationToken>>,
    _subscriptions: Vec<Subscription>,
}

impl AuthorizedPngChooser {
    fn new(workspace: WeakEntity<Workspace>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        Self::new_for(ChooserKind::InputPng, workspace, window, cx)
    }

    fn new_for(
        kind: ChooserKind,
        workspace: WeakEntity<Workspace>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let search_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text(kind.placeholder(), window, cx);
            editor
        });
        let subscription = cx.subscribe(&search_editor, |chooser, _editor, event, cx| {
            if matches!(event, editor::EditorEvent::BufferEdited) && !chooser.opening {
                chooser.offset = 0;
                chooser.previous_offsets.clear();
                chooser.query(cx);
            }
        });
        let mut chooser = Self {
            kind,
            workspace,
            search_editor,
            choices: Vec::new(),
            selected_index: None,
            scroll_handle: ScrollHandle::new(),
            offset: 0,
            previous_offsets: Vec::new(),
            next_offset: None,
            query_generation: 0,
            loading: false,
            opening: false,
            dismissed: false,
            diagnostic: None,
            query_task: None,
            open_task: None,
            open_cancellation: None,
            _subscriptions: vec![subscription],
        };
        chooser.query(cx);
        chooser
    }

    fn query(&mut self, cx: &mut Context<Self>) {
        if self.opening || self.dismissed {
            return;
        }
        self.query_task = None;
        self.choices.clear();
        self.selected_index = None;
        self.next_offset = None;
        self.diagnostic = None;
        let Some(generation) = self.query_generation.checked_add(1) else {
            self.loading = false;
            self.diagnostic = Some("Close and reopen the chooser to continue.".into());
            cx.notify();
            return;
        };
        self.query_generation = generation;
        let search = self.search_editor.read(cx).text(cx);
        let search = search.trim();
        let text = (!search.is_empty()).then(|| search.to_owned());
        match self.kind {
            ChooserKind::InputPng => match crate::query_authorized_pngs(self.offset, text, cx) {
                Ok(query) => self.track_query(
                    generation,
                    async move {
                        query
                            .await
                            .map(|page| ChoicePage {
                                choices: page.choices.into_iter().map(AssetChoice::Png).collect(),
                                next_offset: page.next_offset,
                            })
                            .map_err(|error| error.to_string())
                    },
                    cx,
                ),
                Err(error) => self.query_failed(error.to_string(), cx),
            },
            ChooserKind::PublishedPng => {
                match crate::query_published_png_versions(self.offset, text, cx) {
                    Ok(query) => self.track_query(
                        generation,
                        async move {
                            query
                                .await
                                .map(|page| ChoicePage {
                                    choices: page
                                        .choices
                                        .into_iter()
                                        .map(|choice| AssetChoice::PublishedPng(Box::new(choice)))
                                        .collect(),
                                    next_offset: page.next_offset,
                                })
                                .map_err(|error| error.to_string())
                        },
                        cx,
                    ),
                    Err(error) => self.query_failed(error.to_string(), cx),
                }
            }
            ChooserKind::SavedSim => {
                match crate::query_saved_sim_documents(self.offset, text, cx) {
                    Ok(query) => self.track_query(
                        generation,
                        async move {
                            query
                                .await
                                .map(|page| ChoicePage {
                                    choices: page
                                        .choices
                                        .into_iter()
                                        .map(AssetChoice::Sim)
                                        .collect(),
                                    next_offset: page.next_offset,
                                })
                                .map_err(|error| error.to_string())
                        },
                        cx,
                    ),
                    Err(error) => self.query_failed(error.to_string(), cx),
                }
            }
        }
    }

    fn query_failed(&mut self, error: String, cx: &mut Context<Self>) {
        self.loading = false;
        self.diagnostic = Some(error);
        cx.notify();
    }

    fn track_query(
        &mut self,
        generation: u64,
        query: impl std::future::Future<Output = Result<ChoicePage, String>> + 'static,
        cx: &mut Context<Self>,
    ) {
        self.loading = true;
        self.query_task = Some(cx.spawn(async move |chooser, cx| {
            let page = query.await;
            match chooser.update(cx, |chooser, cx| {
                if chooser.dismissed || chooser.query_generation != generation {
                    return;
                }
                chooser.loading = false;
                match page {
                    Ok(page) => {
                        chooser.choices = page.choices;
                        chooser.selected_index = (!chooser.choices.is_empty()).then_some(0);
                        chooser.scroll_handle.scroll_to_item(0);
                        chooser.next_offset = page.next_offset;
                    }
                    Err(error) => chooser.diagnostic = Some(error),
                }
                cx.notify();
            }) {
                Ok(()) => {}
                Err(error) => log::debug!("Graphite asset chooser query owner released: {error}"),
            }
        }));
        cx.notify();
    }

    fn open(&mut self, choice: AssetChoice, window: &mut Window, cx: &mut Context<Self>) {
        if self.loading || self.opening || self.dismissed {
            return;
        }
        self.cancel_open();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        let opening = match self.workspace.update(cx, |_workspace, cx| match choice {
            AssetChoice::Png(choice) => {
                crate::open_authorized_png_artboard(choice, cancellation.clone(), window, cx)
            }
            AssetChoice::Sim(choice) => {
                crate::open_saved_sim_document(choice, cancellation.clone(), window, cx)
            }
            AssetChoice::PublishedPng(choice) => {
                crate::open_published_png_artboard(*choice, cancellation.clone(), window, cx)
            }
        }) {
            Ok(Ok(opening)) => opening,
            Ok(Err(error)) => {
                cancellation.cancel();
                self.diagnostic = Some(error.to_string());
                cx.notify();
                return;
            }
            Err(error) => {
                cancellation.cancel();
                self.diagnostic = Some(format!("The workspace is no longer available: {error}"));
                cx.notify();
                return;
            }
        };
        self.diagnostic = None;
        self.opening = true;
        self.open_cancellation = Some(cancellation);
        self.query_task = None;
        self.open_task = Some(cx.spawn_in(window, async move |chooser, cx| {
            let result = opening.await;
            match chooser.update_in(cx, |chooser, _window, cx| {
                chooser.opening = false;
                match result {
                    Ok(_item) => {
                        chooser.open_cancellation = None;
                        cx.emit(DismissEvent);
                    }
                    Err(error) => {
                        if let Some(cancellation) = chooser.open_cancellation.take() {
                            cancellation.cancel();
                        }
                        chooser.diagnostic = Some(error.to_string());
                        cx.notify();
                    }
                }
            }) {
                Ok(()) => {}
                Err(error) => log::debug!("Graphite asset chooser open owner released: {error}"),
            }
        }));
        cx.notify();
    }

    fn select(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.loading || self.opening || self.dismissed || index >= self.choices.len() {
            return;
        }
        self.selected_index = Some(index);
        self.scroll_handle.scroll_to_item(index);
        cx.notify();
    }

    fn select_next(&mut self, cx: &mut Context<Self>) {
        let count = self.choices.len();
        if count > 0 {
            let index = match self.selected_index {
                Some(index) if index < count - 1 => index + 1,
                _ => 0,
            };
            self.select(index, cx);
        }
    }

    fn select_previous(&mut self, cx: &mut Context<Self>) {
        let count = self.choices.len();
        if count > 0 {
            let index = match self.selected_index {
                Some(index) if index > 0 && index < count => index - 1,
                _ => count - 1,
            };
            self.select(index, cx);
        }
    }

    fn cancel_open(&mut self) {
        if let Some(cancellation) = self.open_cancellation.take() {
            cancellation.cancel();
        }
        self.open_task = None;
    }

    fn dismiss(&mut self, cx: &mut Context<Self>) {
        self.dismissed = true;
        self.query_task = None;
        self.cancel_open();
        cx.emit(DismissEvent);
    }
}

impl Drop for AuthorizedPngChooser {
    fn drop(&mut self) {
        self.cancel_open();
    }
}

impl ModalView for AuthorizedPngChooser {
    fn on_before_dismiss(
        &mut self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> DismissDecision {
        self.dismissed = true;
        self.query_task = None;
        self.cancel_open();
        DismissDecision::Dismiss(true)
    }
}

impl EventEmitter<DismissEvent> for AuthorizedPngChooser {}

impl Focusable for AuthorizedPngChooser {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.search_editor.focus_handle(cx)
    }
}

impl Render for AuthorizedPngChooser {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let generation = self.query_generation;
        let kind = self.kind;
        let busy = self.loading || self.opening;
        v_flex()
            .id("graphite-png-chooser")
            .debug_selector(move || kind.selector("CHOOSER"))
            .key_context("GraphitePngChooser")
            .track_focus(&self.focus_handle(cx))
            .w(rems(36.))
            .p_3()
            .gap_2()
            .on_action(cx.listener(|chooser, _: &menu::Cancel, _window, cx| {
                chooser.dismiss(cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &editor::actions::Cancel, _window, cx| {
                chooser.dismiss(cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &menu::Confirm, window, cx| {
                if let Some(choice) = chooser
                    .selected_index
                    .and_then(|index| chooser.choices.get(index))
                    .cloned()
                {
                    chooser.open(choice, window, cx);
                }
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &menu::SelectNext, _window, cx| {
                chooser.select_next(cx);
                cx.stop_propagation();
            }))
            .on_action(
                cx.listener(|chooser, _: &menu::SelectPrevious, _window, cx| {
                    chooser.select_previous(cx);
                    cx.stop_propagation();
                }),
            )
            .on_action(cx.listener(|chooser, _: &zed_actions::editor::MoveDown, _window, cx| {
                chooser.select_next(cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &zed_actions::editor::MoveUp, _window, cx| {
                chooser.select_previous(cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &menu::SelectFirst, _window, cx| {
                chooser.select(0, cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|chooser, _: &menu::SelectLast, _window, cx| {
                if let Some(index) = chooser.choices.len().checked_sub(1) {
                    chooser.select(index, cx);
                }
                cx.stop_propagation();
            }))
            .child(Label::new(kind.title()))
            .child(
                Label::new(match kind {
                    ChooserKind::InputPng => "Choose a PNG from the active profile's authorized input assets.",
                    ChooserKind::SavedSim => "Choose saved .sim artwork from the active profile's authorized output assets.",
                    ChooserKind::PublishedPng => "Choose an immutable PNG version from the active profile's published output assets.",
                })
                    .color(Color::Muted),
            )
            .child(
                div()
                    .id("graphite-png-search")
                    .debug_selector(move || kind.selector("SEARCH"))
                    .aria_label(kind.placeholder())
                    .child(self.search_editor.clone()),
            )
            .when(self.loading, |view| {
                view.child(
                    div()
                        .debug_selector(move || kind.selector("LOADING"))
                        .child(Label::new("Loading assets…")),
                )
            })
            .when(self.opening, |view| {
                view.child(
                    div()
                        .debug_selector(move || kind.selector("OPENING"))
                        .child(Label::new("Opening artwork…")),
                )
            })
            .when(
                !busy && self.choices.is_empty() && self.diagnostic.is_none(),
                |view| {
                    view.child(div().debug_selector(move || kind.selector("EMPTY")).child(
                        Label::new(match (kind, self.next_offset.is_some() || self.offset > 0) {
                            (ChooserKind::InputPng, true) => "No PNGs on this page.",
                            (ChooserKind::InputPng, false) => "No authorized PNGs match this search.",
                            (ChooserKind::SavedSim, true) => "No saved .sim documents on this page.",
                            (ChooserKind::SavedSim, false) => "No authorized saved .sim documents match this search.",
                            (ChooserKind::PublishedPng, true) => "No published PNG versions on this page.",
                            (ChooserKind::PublishedPng, false) => "No authorized published PNG versions match this search.",
                        }),
                    ))
                },
            )
            .when_some(self.diagnostic.clone(), |view, diagnostic| {
                view.child(
                    ui::diagnostic_alert(
                        "graphite-png-open-error",
                        diagnostic.clone(),
                        Label::new(diagnostic).color(Color::Error),
                    )
                    .debug_selector(move || kind.selector("OPEN-ERROR")),
                )
            })
            .child(
                v_flex()
                    .id("graphite-png-choices")
                    .role(gpui::Role::ListBox)
                    .aria_label(match kind { ChooserKind::InputPng => "Authorized PNGs", ChooserKind::SavedSim => "Saved Graphite documents", ChooserKind::PublishedPng => "Published PNG versions" })
                    .track_scroll(&self.scroll_handle)
                    .max_h(rems(20.))
                    .overflow_y_scroll()
                    .children(
                        self.choices
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(index, choice)| {
                                let selected = self.selected_index == Some(index);
                                let label = choice.label();
                                let byte_size = choice.byte_size();
                                div()
                                    .debug_selector(move || kind.selector(&format!("ASSET-{index}")))
                                    .child(
                                        ListItem::new(("graphite-png-asset", index))
                                            .disabled(busy)
                                            .aria_role(gpui::Role::ListBoxOption)
                                            .aria_label(format!("Open {label}"))
                                            .toggle_state(selected)
                                            .when(selected, |item| item.aria_active_descendant())
                                            .child(Label::new(label))
                                            .child(
                                                Label::new(format!("{byte_size} bytes"))
                                                    .color(Color::Muted),
                                            )
                                            .on_click(cx.listener(
                                                move |chooser, _event, window, cx| {
                                                    if chooser.query_generation == generation {
                                                        chooser.open(choice.clone(), window, cx);
                                                    }
                                                },
                                            )),
                                    )
                            }),
                    ),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .debug_selector(move || kind.selector("PREVIOUS"))
                            .child(
                                Button::new("graphite-png-previous", "Previous")
                                    .disabled(busy || self.previous_offsets.is_empty())
                                    .on_click(cx.listener(|chooser, _event, _window, cx| {
                                        if let Some(offset) = chooser.previous_offsets.pop() {
                                            chooser.offset = offset;
                                            chooser.query(cx);
                                        }
                                    })),
                            ),
                    )
                    .child(
                        div().debug_selector(move || kind.selector("NEXT")).child(
                            Button::new("graphite-png-next", "Next")
                                .disabled(busy || self.next_offset.is_none())
                                .on_click(cx.listener(|chooser, _event, _window, cx| {
                                    if let Some(offset) = chooser.next_offset {
                                        chooser.previous_offsets.push(chooser.offset);
                                        chooser.offset = offset;
                                        chooser.query(cx);
                                    }
                                })),
                        ),
                    )
                    .child(
                        div().debug_selector(move || kind.selector("RETRY")).child(
                            Button::new(
                                "graphite-png-retry",
                                if self.diagnostic.is_some() {
                                    "Retry"
                                } else {
                                    "Refresh"
                                },
                            )
                            .disabled(busy)
                            .on_click(
                                cx.listener(|chooser, _event, _window, cx| chooser.query(cx)),
                            ),
                        ),
                    )
                    .child(div().debug_selector(move || kind.selector("CANCEL")).child(
                        Button::new("graphite-png-cancel", "Cancel").on_click(
                            cx.listener(|chooser, _event, _window, cx| chooser.dismiss(cx)),
                        ),
                    )),
            )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::graph::GraphiteArtboardItem;
    use crate::resources::tests::{host_reader_fixture, indexed_request_png};
    use gpui::{TestAppContext, VisualTestContext};
    use project::{FakeFs, Project};
    use std::path::Path;

    pub(crate) async fn native_workspace(
        cx: &mut TestAppContext,
    ) -> (Entity<Workspace>, &mut VisualTestContext) {
        cx.update(|cx| {
            assets::Assets.load_test_fonts(cx);
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            cx.set_global(db::AppDatabase::test_new());
            let state = workspace::AppState::test(cx);
            workspace::AppState::set_global(state, cx);
            crate::init(cx);
        });
        let filesystem = FakeFs::new(cx.executor());
        filesystem
            .insert_tree(Path::new("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(filesystem, [Path::new("/project")], cx).await;
        let (host, window) = cx
            .add_window_view(|window, cx| workspace::MultiWorkspace::test_new(project, window, cx));
        let workspace = host.read_with(window, |host, _| host.workspace().clone());
        window.update(|_, cx| {
            let bindings = settings::KeymapFile::load_asset_allow_partial_failure(
                "keymaps/default-linux.json",
                cx,
            )
            .expect("actual application keymap");
            cx.bind_keys(bindings);
        });
        window.simulate_resize(gpui::size(gpui::px(1400.), gpui::px(1400.)));
        window.run_until_parked();
        paint(window);
        (workspace, window)
    }

    pub(crate) fn register_fixture_workflow(
        fixture: &crate::resources::tests::HostReaderFixture,
        png: &crate::resources::tests::IndexedRequestPng,
        window: &mut VisualTestContext,
    ) -> (
        sim_runtime::AuthorizedCapabilities,
        sim_runtime::SharedOutputCommitter,
        sim_runtime::AuthorizedCapabilities,
    ) {
        let existing = window.update(|_, cx| {
            crate::asset_binding(cx).and_then(|binding| {
                match binding.update(cx, |binding, cx| binding.project_context(cx)) {
                    Ok(project) if std::sync::Arc::ptr_eq(&project.assets, &fixture.assets) => {
                        Some(project)
                    }
                    Ok(_) => None,
                    Err(error) => {
                        log::debug!("Fixture publisher has not been registered: {error}");
                        None
                    }
                }
            })
        });
        if let Some(project) = existing {
            return (project.output_reader, project.committer, project.publisher);
        }
        let (_, _, publisher_authorization, output_reader) =
            crate::interop::tests::output_reader_tests::published(
                fixture,
                png,
                &uuid::Uuid::new_v4().to_string(),
                "chooser-policy-fixture",
            );
        let roots = fixture
            .assets
            .lock()
            .expect("actual asset service")
            .roots()
            .clone();
        let committer = std::sync::Arc::new(std::sync::Mutex::new(
            sim_runtime::OutputCommitter::open(roots).expect("actual canonical publisher"),
        ));
        window.update(|_, cx| {
            crate::register_asset_binding_with_output_reader(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                Some(output_reader.clone()),
                cx,
            )
            .expect("genuine Input and Output readers");
            crate::register_asset_publisher(committer.clone(), publisher_authorization.clone(), cx)
                .expect("same-root genuine publisher authority");
        });
        (output_reader, committer, publisher_authorization)
    }

    pub(crate) fn paint(window: &mut VisualTestContext) {
        window.update(|window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        });
    }

    fn click_without_pumping(window: &mut VisualTestContext, selector: &'static str) {
        let position = window
            .debug_bounds(selector)
            .expect("visible native PNG control")
            .center();
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::MouseDown(gpui::MouseDownEvent {
                    position,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
                cx,
            );
            window.dispatch_event(
                gpui::PlatformInput::MouseUp(gpui::MouseUpEvent {
                    position,
                    button: gpui::MouseButton::Left,
                    modifiers: gpui::Modifiers::default(),
                    click_count: 1,
                }),
                cx,
            );
        });
    }

    pub(crate) fn click(window: &mut VisualTestContext, selector: &'static str) {
        click_without_pumping(window, selector);
        window.run_until_parked();
        paint(window);
    }

    fn chooser(
        workspace: &Entity<Workspace>,
        window: &VisualTestContext,
    ) -> Entity<AuthorizedPngChooser> {
        workspace
            .read_with(window, |workspace, cx| {
                workspace.active_modal::<AuthorizedPngChooser>(cx)
            })
            .expect("registered action opened the real Workspace modal")
    }

    fn search_requested(modal: &Entity<AuthorizedPngChooser>, window: &mut VisualTestContext) {
        let editor = modal.read_with(window, |modal, _| modal.search_editor.clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input("requested.png");
        window.run_until_parked();
        paint(window);
        modal.read_with(window, |modal, _| {
            assert!(!modal.loading);
            assert!(!modal.opening);
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
            assert!(
                modal
                    .choices
                    .first()
                    .expect("indexed PNG")
                    .label()
                    .contains("requested.png")
            );
        });
    }

    fn edit_artboard_field(
        item: &Entity<GraphiteArtboardItem>,
        index: usize,
        text: &str,
        window: &mut VisualTestContext,
    ) {
        let editor = item.read_with(window, |item, _| {
            item.artboard_editors
                .get(index)
                .expect("existing native field")
                .clone()
        });
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
    }

    pub(crate) fn open_requested_png(
        workspace: &Entity<Workspace>,
        window: &mut VisualTestContext,
    ) -> Entity<GraphiteArtboardItem> {
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        search_requested(&chooser(workspace, window), window);
        click(window, "GRAPHITE-PNG-ASSET-0");
        workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("authorized native PNG Item")
    }

    pub(crate) fn output_records(
        fixture: &crate::resources::tests::HostReaderFixture,
        reader: &sim_runtime::AuthorizedCapabilities,
        text: &str,
    ) -> Vec<sim_runtime::assets::AssetRecord> {
        fixture
            .assets
            .lock()
            .expect("actual shared catalog")
            .list_authorized(
                &sim_runtime::assets::AssetQuery {
                    namespace: Some(sim_runtime::assets::AssetNamespace::Output),
                    text: Some(text.into()),
                    ..Default::default()
                },
                reader,
            )
            .expect("actual authorized catalog page")
            .records
    }

    pub(crate) async fn close_artboard(
        workspace: &Entity<Workspace>,
        item: Entity<GraphiteArtboardItem>,
        window: &mut VisualTestContext,
    ) {
        let item_id = item.entity_id();
        workspace
            .update_in(window, |workspace, window, cx| {
                workspace.active_pane().clone().update(cx, |pane, cx| {
                    pane.close_item_by_id(item_id, workspace::SaveIntent::Close, window, cx)
                })
            })
            .await
            .expect("actual native close/save route");
        let weak = item.downgrade();
        drop(item);
        paint(window);
        window.run_until_parked();
        paint(window);
        assert!(weak.upgrade().is_none());
    }

    fn saved_sim_search(
        workspace: &Entity<Workspace>,
        text: &str,
        window: &mut VisualTestContext,
    ) -> Entity<AuthorizedPngChooser> {
        window.dispatch_action(OpenSimDocument);
        window.run_until_parked();
        paint(window);
        let modal = chooser(workspace, window);
        let search = modal.read_with(window, |modal, _| modal.search_editor.clone());
        search.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
        paint(window);
        modal
    }

    #[gpui::test]
    async fn at_r09_r08_png_action_opens_authorized_source_and_real_native_preview(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real host asset service");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed PNG producer");
        let source_bytes = png.encoded.clone();
        let (workspace, window) = native_workspace(cx).await;
        let (output_reader, _, _) = register_fixture_workflow(&fixture, &png, window);
        let workspace_focus =
            workspace.read_with(window, |workspace, cx| workspace.focus_handle(cx));
        window.update(|window, cx| workspace_focus.focus(window, cx));
        paint(window);
        window.update(|window, cx| {
            assert!(window.available_actions(cx).iter().any(|action| action.name() == <OpenPngEditor as gpui::Action>::name_for_type()));
        });
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-PNG-CHOOSER").is_some());
        let modal = chooser(&workspace, window);
        search_requested(&modal, window);
        assert_eq!(
            modal.read_with(window, |modal, _| modal
                .choices
                .first()
                .expect("genuine choice")
                .byte_size()),
            png.encoded.len() as u64
        );
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        click(window, "GRAPHITE-PNG-ASSET-0");
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("authorized facade installed the existing native Artboard Item");
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        let session = item.read_with(window, |item, _| item.session().clone());
        let input_version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("real pinned project input")
                .clone()
        });
        let resolved = fixture
            .assets
            .lock()
            .expect("actual service")
            .resolve_project_asset_version(
                &input_version,
                &output_reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("genuine immutable input version");
        assert_eq!(resolved.version().original_identity, png.asset_identity);
        assert_eq!(resolved.bytes(), source_bytes);
        let source = session.read_with(window, |session, _| {
            assert!(!session.resource_is_revoked());
            assert!(!session.is_pending());
            assert!(
                session
                    .preview_is_current(session.preview().expect("chooser opt-in initial preview"))
            );
            assert_eq!(
                session.configuration().viewport.resolution.to_array(),
                [1920_u32, 1080_u32]
            );
            let source = session
                .document()
                .encoded_png_input()
                .expect("original encoded document")
                .source()
                .clone();
            assert_eq!(
                source.identity().asset_id.as_ref(),
                input_version.asset_id.to_string()
            );
            assert_eq!(
                source.identity().content_sha256,
                png.image_identity.content_sha256
            );
            assert_eq!((source.width(), source.height()), (3, 1));
            source
        });
        session.read_with(window, |session, _| {
            let preview = session
                .preview()
                .expect("automatic initial native preview completed");
            assert!(session.preview_is_current(preview));
            let resolution = session.configuration().viewport.resolution;
            assert_eq!(
                (
                    preview.image().size(0).width.0,
                    preview.image().size(0).height.0
                ),
                (
                    i32::try_from(resolution.x).expect("bounded native scene width"),
                    i32::try_from(resolution.y).expect("bounded native scene height")
                )
            );
            let pixels = preview
                .image()
                .as_bytes(0)
                .expect("actual native BGRA frame");
            assert_eq!(
                pixels.len(),
                resolution.x as usize * resolution.y as usize * 4
            );
            assert!(pixels.as_chunks::<4>().0.iter().any(|pixel| pixel[3] > 0));
            assert!(
                pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[3] > 0 && pixel[..3].iter().any(|channel| *channel < 250))
            );
            assert!(std::sync::Arc::ptr_eq(
                preview
                    .output()
                    .encoded_png_input()
                    .expect("captured authentic input")
                    .source(),
                &source
            ));
            assert_eq!(
                preview
                    .output()
                    .output()
                    .metadata
                    .clip_targets
                    .contains(&session.document().node_id()),
                session.parameters().clip
            );
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let prepared = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("native prepared PNG action")
                .clone()
        });
        let (initial, revision) = session.read_with(window, |session, _| {
            assert!(session.png_is_current(&prepared));
            (
                session.preview().expect("initial preview").clone(),
                session.revision(),
            )
        });
        let initial_frame = initial
            .image()
            .as_bytes(0)
            .expect("initial full frame")
            .to_vec();
        let editor = item.read_with(window, |item, _| item.artboard_editors[9].clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input("12");
        window.run_until_parked();
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision)
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision + 1);
            assert_eq!(
                session
                    .image_placement()
                    .expect("native committed placement")
                    .translation
                    .x
                    .to_bits(),
                12_f64.to_bits()
            );
            let current = session
                .preview()
                .expect("automatic refresh after committed property");
            assert!(session.preview_is_current(current));
            assert!(!session.preview_is_current(&initial));
            assert!(!session.png_is_current(&prepared));
            assert_ne!(
                current
                    .image()
                    .as_bytes(0)
                    .expect("current real native frame"),
                initial_frame
            );
            assert!(std::sync::Arc::ptr_eq(
                current
                    .output()
                    .encoded_png_input()
                    .expect("retained source after edit")
                    .source(),
                &source
            ));
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let edited = item.read_with(window, |item, _| {
            item.prepared_png().expect("new prepared PNG").clone()
        });
        session.read_with(window, |session, _| {
            assert!(session.png_is_current(&edited))
        });
        assert_ne!(edited.pixels_sha256(), prepared.pixels_sha256());
        assert_ne!(edited.png().encoded_bytes(), prepared.png().encoded_bytes());
        assert_eq!(png.encoded, source_bytes);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual service")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("source still verifies after native preview"),
            source_bytes
        );
    }

    #[gpui::test]
    async fn at_r09_r08_png_chooser_reports_missing_and_stale_binding_then_retries(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real host service");
        let png = indexed_request_png(&fixture, 3).expect("genuine indexed PNG");
        let (workspace, window) = native_workspace(cx).await;
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let modal = chooser(&workspace, window);
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_some());
        modal.read_with(window, |modal, _| {
            assert!(!modal.loading);
            assert!(modal.choices.is_empty());
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        register_fixture_workflow(&fixture, &png, window);
        click(window, "GRAPHITE-PNG-RETRY");
        search_requested(&modal, window);
        window.update(|_window, cx| crate::clear_asset_binding(cx).expect("actual host shutdown"));
        click(window, "GRAPHITE-PNG-ASSET-0");
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_some());
        modal.read_with(window, |modal, _| {
            assert!(!modal.opening);
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        register_fixture_workflow(&fixture, &png, window);
        click(window, "GRAPHITE-PNG-RETRY");
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_none());
        click(window, "GRAPHITE-PNG-ASSET-0");
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteArtboardItem>(cx)
                .is_some()
        }));
    }

    #[gpui::test]
    async fn at_r09_r08_png_chooser_pending_dismissal_cannot_install_a_late_item(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real authorized service");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed PNG");
        let (workspace, window) = native_workspace(cx).await;
        register_fixture_workflow(&fixture, &png, window);
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let modal = chooser(&workspace, window);
        search_requested(&modal, window);
        let weak_modal = modal.downgrade();
        click_without_pumping(window, "GRAPHITE-PNG-ASSET-0");
        let cancellation = modal.read_with(window, |modal, _| {
            modal
                .open_cancellation
                .as_ref()
                .expect("caller-owned installation token")
                .clone()
        });
        modal.read_with(window, |modal, _| {
            assert!(modal.opening);
            assert!(modal.open_task.is_some());
        });
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        paint(window);
        click_without_pumping(window, "GRAPHITE-PNG-CANCEL");
        assert!(cancellation.is_cancelled());
        drop(modal);
        paint(window);
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(weak_modal.upgrade().is_none());
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let retry = chooser(&workspace, window);
        search_requested(&retry, window);
        click(window, "GRAPHITE-PNG-ASSET-0");
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace
                .active_item_as::<GraphiteArtboardItem>(cx)
                .is_some()
        }));
    }

    #[gpui::test]
    async fn at_r09_r08_png_chooser_keyboard_selects_second_result_and_retries(
        cx: &mut TestAppContext,
    ) {
        use sim_runtime::assets::{AssetNamespace, UploadRequest};
        use sim_runtime::{
            AssetOperation, Capability, CapabilitySet, PermissionGrant, PermissionPolicy,
        };
        let fixture = host_reader_fixture().expect("real host service");
        let png = indexed_request_png(&fixture, 4).expect("real indexed PNG");
        let capabilities = CapabilitySet::new([Capability::Asset {
            namespace: AssetNamespace::Input.locator_type().to_owned(),
            action: AssetOperation::Write,
        }]);
        let grant = PermissionGrant::new(
            &png.asset_identity.profile_id,
            "keyboard-chooser-fixture",
            capabilities.clone(),
            "isolated-local-asset-test",
        )
        .expect("actual fixture policy grant");
        let authorization = PermissionPolicy::new(&png.asset_identity.profile_id, [grant])
            .expect("actual fixture policy")
            .authorize("keyboard-chooser-fixture", &capabilities)
            .expect("fixture-only producer authorization");
        let second = fixture
            .assets
            .lock()
            .expect("real service")
            .upload(
                UploadRequest {
                    namespace: AssetNamespace::Input,
                    filename: "second.png".into(),
                    subfolder: Default::default(),
                    bytes: png.encoded.clone(),
                    overwrite: false,
                    tags: Default::default(),
                },
                &authorization,
                &sim_tensor::CancellationToken::default(),
            )
            .expect("actual indexed second PNG producer")
            .record;
        let (workspace, window) = native_workspace(cx).await;
        let (output_reader, _, _) = register_fixture_workflow(&fixture, &png, window);
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let modal = chooser(&workspace, window);
        let editor = modal.read_with(window, |modal, _| modal.search_editor.clone());
        editor.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(".png");
        window.run_until_parked();
        paint(window);
        modal.read_with(window, |modal, _| {
            assert_eq!(modal.choices.len(), 2);
            assert_eq!(modal.selected_index, Some(0));
            assert!(modal.choices[0].label().contains("requested.png"));
            assert!(modal.choices[1].label().contains("second.png"));
        });
        window.simulate_keystrokes("down");
        window.run_until_parked();
        paint(window);
        modal.read_with(window, |modal, _| assert_eq!(modal.selected_index, Some(1)));
        window.simulate_keystrokes("up");
        window.run_until_parked();
        modal.read_with(window, |modal, _| assert_eq!(modal.selected_index, Some(0)));
        window.simulate_keystrokes("ctrl-p");
        window.run_until_parked();
        modal.read_with(window, |modal, _| assert_eq!(modal.selected_index, Some(1)));
        window.simulate_keystrokes("ctrl-n");
        window.run_until_parked();
        modal.read_with(window, |modal, _| assert_eq!(modal.selected_index, Some(0)));
        window.simulate_keystrokes("down");
        window.run_until_parked();
        window.update(|_, cx| crate::clear_asset_binding(cx).expect("actual reader retirement"));
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_some());
        modal.read_with(window, |modal, _| {
            assert!(!modal.opening);
            assert_eq!(modal.selected_index, Some(1));
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        register_fixture_workflow(&fixture, &png, window);
        click(window, "GRAPHITE-PNG-RETRY");
        modal.read_with(window, |modal, _| {
            assert_eq!(modal.choices.len(), 2);
            assert_eq!(modal.selected_index, Some(0));
            assert!(modal.diagnostic.is_none());
        });
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_none());
        let search_focus = modal.read_with(window, |modal, cx| modal.focus_handle(cx));
        window.update(|window, cx| search_focus.focus(window, cx));
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        register_fixture_workflow(&fixture, &png, window);
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let retry = chooser(&workspace, window);
        let search = retry.read_with(window, |modal, _| modal.search_editor.clone());
        search.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(".png");
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_none());
        window.simulate_keystrokes("down enter");
        window.run_until_parked();
        paint(window);
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("native Enter installed the selected second PNG");
        let input_version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("actual pinned selected version")
                .clone()
        });
        let resolved = fixture
            .assets
            .lock()
            .expect("actual service")
            .resolve_project_asset_version(
                &input_version,
                &output_reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("selected immutable project input");
        assert_eq!(resolved.version().original_identity, second.identity);
        assert_eq!(resolved.bytes(), png.encoded);
        let session = item.read_with(window, |item, _| item.session().clone());
        session.read_with(window, |session, _| {
            let source = session
                .document()
                .encoded_png_input()
                .expect("genuine captured source")
                .source();
            assert_eq!(
                source.identity().asset_id.as_ref(),
                input_version.asset_id.to_string()
            );
            assert_eq!(
                source.identity().content_sha256,
                png.image_identity.content_sha256
            );
            let preview = session.preview().expect("real automatic preview");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                (
                    preview.image().size(0).width.0,
                    preview.image().size(0).height.0
                ),
                (1920_i32, 1080_i32)
            );
            assert!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("actual native frame")
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[3] > 0 && pixel[..3].iter().any(|channel| *channel < 250))
            );
        });
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual service")
                .read_verified(
                    &second.identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("selected source remains immutable"),
            png.encoded
        );
    }

    #[gpui::test]
    async fn at_task573_r09_saved_sim_action_is_discoverable_and_reports_authorized_reader_errors(
        cx: &mut TestAppContext,
    ) {
        let (workspace, window) = native_workspace(cx).await;
        let focus = workspace.read_with(window, |workspace, cx| workspace.focus_handle(cx));
        window.update(|window, cx| focus.focus(window, cx));
        paint(window);
        window.update(|window, cx| {
            for action_name in [
                <OpenPngEditor as gpui::Action>::name_for_type(),
                <OpenSimDocument as gpui::Action>::name_for_type(),
            ] {
                assert!(
                    window
                        .available_actions(cx)
                        .iter()
                        .any(|action| action.name() == action_name)
                );
            }
        });
        window.dispatch_action(OpenSimDocument);
        window.run_until_parked();
        paint(window);
        let modal = chooser(&workspace, window);
        assert!(window.debug_bounds("GRAPHITE-SIM-CHOOSER").is_some());
        assert!(window.debug_bounds("GRAPHITE-SIM-SEARCH").is_some());
        assert!(window.debug_bounds("GRAPHITE-SIM-OPEN-ERROR").is_some());
        assert!(window.debug_bounds("GRAPHITE-PNG-CHOOSER").is_none());
        modal.read_with(window, |modal, _| {
            assert!(matches!(modal.kind, ChooserKind::SavedSim));
            assert!(!modal.loading);
            assert!(!modal.opening);
            assert!(modal.choices.is_empty());
            assert!(modal.selected_index.is_none());
            assert!(modal.open_cancellation.is_none());
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|error| !error.is_empty())
            );
        });
        let search = modal.read_with(window, |modal, _| modal.search_editor.clone());
        search.update_in(window, |editor, window, cx| {
            editor.focus_handle(cx).focus(window, cx)
        });
        window.simulate_input("saved.sim");
        window.run_until_parked();
        paint(window);
        let generation = modal.read_with(window, |modal, _| modal.query_generation);
        click(window, "GRAPHITE-SIM-RETRY");
        modal.read_with(window, |modal, cx| {
            assert_eq!(modal.query_generation, generation + 1);
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|error| !error.is_empty())
            );
            assert_eq!(modal.search_editor.read(cx).text(cx), "saved.sim");
        });
        assert!(window.debug_bounds("GRAPHITE-SIM-OPEN-ERROR").is_some());
        let search_focus = modal.read_with(window, |modal, cx| modal.focus_handle(cx));
        window.update(|window, cx| search_focus.focus(window, cx));
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_some()
        }));
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-PNG-CHOOSER").is_some());
        window.dispatch_action(OpenSimDocument);
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-SIM-CHOOSER").is_some());
        assert!(window.debug_bounds("GRAPHITE-PNG-CHOOSER").is_none());
        chooser(&workspace, window).read_with(window, |modal, cx| {
            assert!(matches!(modal.kind, ChooserKind::SavedSim));
            assert_eq!(modal.search_editor.read(cx).text(cx), "");
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|error| !error.is_empty())
            );
        });
        click(window, "GRAPHITE-SIM-CANCEL");
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
    }

    #[gpui::test]
    async fn at_task573_png_workflow_saves_reopens_immutable_source_and_publishes_edited_png(
        cx: &mut TestAppContext,
    ) {
        use sha2::{Digest as _, Sha256};
        let fixture = host_reader_fixture().expect("real shared project service");
        let png = indexed_request_png(&fixture, 4).expect("actual Input producer");
        assert_eq!(
            fixture.input_authorization.profile_id(),
            "53c3ea98-67d5-4d96-8817-c91236585024"
        );
        let original_bytes = png.encoded.clone();
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, writer) = register_fixture_workflow(&fixture, &png, window);
        let item = open_requested_png(&workspace, window);
        let session = item.read_with(window, |item, _| item.session().clone());
        let input_version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("authentic project version")
                .clone()
        });
        assert_eq!(
            input_version.content_sha256,
            format!("{:x}", Sha256::digest(&original_bytes))
        );
        let pinned = fixture
            .assets
            .lock()
            .expect("actual service")
            .resolve_project_asset_version(
                &input_version,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("actual immutable version");
        assert_eq!(pinned.bytes(), original_bytes);
        assert_eq!(pinned.version().original_identity, png.asset_identity);
        edit_artboard_field(&item, 9, "12", window);
        edit_artboard_field(&item, 10, "8", window);
        let (
            parameters,
            configuration,
            placement,
            revision,
            network,
            document_encoding,
            node_id,
            saved_frame,
        ) = session.read_with(window, |session, _| {
            let preview = session.preview().expect("current automatic original scene");
            assert!(session.preview_is_current(preview));
            let placement = session.image_placement().expect("editable original affine");
            assert_eq!(placement.translation.to_array(), [12., 8.]);
            (
                session.parameters(),
                session.configuration(),
                placement,
                session.revision(),
                session.document().network().clone(),
                session.document().document_id_encoding(),
                session.document().node_id(),
                preview
                    .image()
                    .as_bytes(0)
                    .expect("exact native frame")
                    .to_vec(),
            )
        });
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-RESULT")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_none()
        );
        let saved = item.read_with(window, |item, cx| {
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Saved .sim:"))
            );
            assert!(item.delivery_diagnostic().is_none());
            item.saved_sim_source(cx)
                .expect("actual saved source receipt")
                .clone()
        });
        let saved_bytes = fixture
            .assets
            .lock()
            .expect("actual catalog")
            .read_verified(
                &saved.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("real saved .sim bytes");
        assert_eq!(
            saved.content_sha256,
            format!("{:x}", Sha256::digest(&saved_bytes))
        );
        let container = sim_runtime::SimDocument::decode(&saved_bytes)
            .expect("real shared container integrity");
        assert_eq!(
            container.project_id.to_string(),
            fixture.input_authorization.profile_id()
        );
        assert_eq!(container.documents.len(), 1);
        let engine = container
            .documents
            .first()
            .expect("authentic serialized Graphite engine");
        assert_eq!(engine.engine, "graphite");
        assert_eq!(
            engine.source_revision,
            graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION
        );
        assert_eq!(
            engine.serialization_format,
            "sim.graphite.legacy-artboard.v1"
        );
        assert_eq!(engine.dependencies, vec![input_version.clone()]);
        assert!(!engine.payload.is_empty());
        assert_eq!(
            engine.payload_sha256,
            format!("{:x}", Sha256::digest(engine.payload.as_bytes()))
        );
        let payload: serde_json::Value =
            serde_json::from_str(&engine.payload).expect("actual original payload wrapper");
        assert!(payload["legacy_document"].is_object());
        assert_eq!(payload["capture"]["revision"], serde_json::json!(revision));
        let document_id = engine.document_id;
        drop(session);
        close_artboard(&workspace, item, window).await;
        let changed = indexed_request_png(&fixture, 3).expect("actual current Input replacement");
        assert_ne!(changed.encoded, original_bytes);
        let modal = saved_sim_search(&workspace, ".sim", window);
        modal.read_with(window, |modal, _| {
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
            assert_eq!(
                modal.choices[0].label(),
                saved.identity.relative_path.to_string_lossy()
            );
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("actual saved .sim opener");
        let session = reopened.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&input_version));
            assert_eq!(item.saved_sim_source(cx), Some(&saved));
            item.session().clone()
        });
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            assert_eq!(
                session
                    .image_placement()
                    .expect("restored affine")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(session.document().document_id_encoding(), document_encoding);
            assert_eq!(session.document().node_id(), node_id);
            assert_eq!(session.document().network(), &network);
            let input = session
                .document()
                .encoded_png_input()
                .expect("restored sealed source")
                .source();
            assert_eq!(
                input.identity().asset_id.as_ref(),
                input_version.asset_id.to_string()
            );
            assert_eq!(
                input.identity().content_sha256,
                png.image_identity.content_sha256
            );
            assert_eq!(
                input.identity().version_id.as_ref(),
                input_version.version_id.to_string()
            );
            let preview = session
                .preview()
                .expect("automatic reopened original preview");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("reopened exact native frame"),
                saved_frame
            );
        });
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual catalog")
                .resolve_project_asset_version(
                    &input_version,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("old immutable bytes after Input changed")
                .bytes(),
            original_bytes
        );
        let old_preview = session.read_with(window, |session, _| {
            session.preview().expect("restored preview").clone()
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let old_png = reopened.read_with(window, |item, _| {
            item.prepared_png()
                .expect("actual committed prepared frame")
                .clone()
        });
        session.read_with(window, |session, _| {
            assert!(session.png_is_current(&old_png))
        });
        click(window, "GRAPHITE-ARTBOARD-UPDATE-INPUT");
        let updated_version = reopened.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            item.input_version(cx)
                .expect("actual explicitly updated input version")
                .clone()
        });
        assert_eq!(updated_version.asset_id, input_version.asset_id);
        assert_ne!(updated_version.version_id, input_version.version_id);
        assert_ne!(updated_version.content_sha256, input_version.content_sha256);
        let updated = fixture
            .assets
            .lock()
            .expect("actual version catalog")
            .resolve_project_asset_version(
                &updated_version,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("new immutable input from explicit native update");
        assert_eq!(updated.bytes(), changed.encoded);
        assert_eq!(updated.version().original_identity, png.asset_identity);
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision + 1);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            assert_eq!(
                session
                    .image_placement()
                    .expect("placement preserved by source update")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert!(!session.preview_is_current(&old_preview));
            assert!(!session.png_is_current(&old_png));
            let preview = session.preview().expect("automatic updated-source scene");
            assert!(session.preview_is_current(preview));
            assert_ne!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("visible changed source frame"),
                saved_frame
            );
            let source = preview
                .output()
                .encoded_png_input()
                .expect("genuine updated source snapshot")
                .source();
            assert_eq!(
                source.identity().content_sha256,
                changed.image_identity.content_sha256
            );
            assert_eq!(
                source.identity().version_id.as_ref(),
                updated_version.version_id.to_string()
            );
        });
        assert_eq!(
            old_png
                .source_output()
                .encoded_png_input()
                .expect("old immutable PNG source snapshot")
                .source()
                .identity()
                .content_sha256,
            png.image_identity.content_sha256
        );
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let unchanged_png = reopened.read_with(window, |item, _| {
            item.prepared_png()
                .expect("prepared updated-source PNG")
                .clone()
        });
        let (unchanged_revision, unchanged_preview, unchanged_source) =
            session.read_with(window, |session, _| {
                assert!(session.png_is_current(&unchanged_png));
                (
                    session.revision(),
                    session
                        .preview()
                        .expect("current updated-source frame")
                        .clone(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("current source")
                        .source()
                        .clone(),
                )
            });
        click(window, "GRAPHITE-ARTBOARD-UPDATE-INPUT");
        reopened.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&updated_version));
            assert_eq!(
                item.delivery_status(),
                Some("The pinned input version is already current")
            );
            assert!(item.delivery_diagnostic().is_none());
            assert_eq!(
                item.prepared_png()
                    .expect("unchanged Item PNG retained")
                    .pixels_sha256(),
                unchanged_png.pixels_sha256()
            );
        });
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), unchanged_revision);
            assert!(session.preview_is_current(&unchanged_preview));
            assert!(session.png_is_current(&unchanged_png));
            assert!(std::sync::Arc::ptr_eq(
                session.preview().expect("same current frame").image(),
                unchanged_preview.image()
            ));
            assert!(std::sync::Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("same retained source")
                    .source(),
                &unchanged_source
            ));
        });
        edit_artboard_field(&reopened, 9, "18", window);
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision + 2);
            assert!(!session.preview_is_current(&old_preview));
            assert!(!session.png_is_current(&old_png));
            assert_eq!(
                session
                    .image_placement()
                    .expect("native reopened edit")
                    .translation
                    .x
                    .to_bits(),
                18_f64.to_bits()
            );
            assert!(session.preview_is_current(session.preview().expect("auto-refreshed edit")));
        });
        assert!(output_records(&fixture, &reader, "/baked").is_empty());
        click_without_pumping(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        paint(window);
        click_without_pumping(window, "GRAPHITE-ARTBOARD-CANCEL-DELIVERY");
        window.run_until_parked();
        paint(window);
        assert!(output_records(&fixture, &reader, "/baked").is_empty());
        reopened.read_with(window, |item, _| {
            assert!(item.delivery_status().is_none());
            assert!(
                item.delivery_diagnostic()
                    .is_some_and(|message| message.contains("cancelled"))
            );
        });
        click_without_pumping(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        session.update(window, |session, cx| {
            let mut replacement = session
                .image_placement()
                .expect("same source current placement");
            replacement.translation.x = 24.;
            assert!(
                session
                    .apply_image_placement(session.revision(), replacement, cx)
                    .expect("real committed successor during pending export")
            );
        });
        window.run_until_parked();
        paint(window);
        assert!(output_records(&fixture, &reader, "/baked").is_empty());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_some()
        );
        reopened.read_with(window, |item, _| {
            assert!(
                item.delivery_diagnostic()
                    .is_some_and(|message| !message.is_empty())
            )
        });
        edit_artboard_field(&reopened, 9, "30", window);
        let current = session.read_with(window, |session, _| {
            let current = session.preview().expect("automatic native retry frame");
            assert!(session.preview_is_current(current));
            assert_eq!(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source retained")
                    .source()
                    .identity()
                    .content_sha256,
                changed.image_identity.content_sha256
            );
            current.clone()
        });
        let mut rgba = current
            .image()
            .as_bytes(0)
            .expect("actual exact BGRA frame")
            .to_vec();
        for pixel in rgba.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        assert_ne!(
            current
                .image()
                .as_bytes(0)
                .expect("changed native composition frame"),
            saved_frame
        );
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-RESULT")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_none()
        );
        let exported_records = output_records(&fixture, &reader, "/baked");
        assert_eq!(exported_records.len(), 1);
        let exported = exported_records
            .first()
            .expect("actual committed baked destination");
        assert_eq!(
            exported.identity.namespace,
            sim_runtime::assets::AssetNamespace::Output
        );
        assert_eq!(
            exported
                .identity
                .relative_path
                .parent()
                .expect("canonical document output folder"),
            Path::new("graphite").join(document_id.to_string())
        );
        assert!(
            exported
                .identity
                .filename()
                .expect("actual published basename")
                .starts_with("baked_")
        );
        let exported_bytes = fixture
            .assets
            .lock()
            .expect("actual catalog")
            .read_verified(
                &exported.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("actual delivered PNG bytes");
        assert_eq!(
            exported.sha256,
            format!("{:x}", Sha256::digest(&exported_bytes))
        );
        let decoded = image::load_from_memory_with_format(&exported_bytes, image::ImageFormat::Png)
            .expect("existing real PNG decoder")
            .into_rgba8();
        assert_eq!(decoded.dimensions(), (1920_u32, 1080_u32));
        assert_eq!(decoded.as_raw(), &rgba);
        let operation = committer
            .lock()
            .expect("actual canonical publication journal")
            .operations()
            .iter()
            .find(|operation| {
                operation.identity == exported.identity
                    && operation.sha256 == exported.sha256
                    && operation.state == sim_runtime::OutputOperationState::Committed
            })
            .expect("actual committed baked publication")
            .clone();
        let reference = sim_runtime::ProjectAssetVersionReference {
            asset_id: document_id,
            version_id: operation.operation_id,
            content_sha256: exported.sha256.clone(),
        };
        let derived = fixture
            .assets
            .lock()
            .expect("actual version catalog")
            .resolve_project_asset_version(
                &reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("genuine published derived version");
        assert_eq!(derived.bytes(), exported_bytes);
        assert_eq!(derived.version().stored_identity, exported.identity);
        assert_eq!(
            derived.version().dependencies,
            vec![updated_version.clone()]
        );
        assert_eq!(
            derived.version().provenance["input_versions"],
            serde_json::json!([updated_version])
        );
        let latest_saved = reopened.read_with(window, |item, cx| {
            assert!(item.delivery_status().is_some_and(|status| {
                status.contains(
                    &exported
                        .identity
                        .to_reference()
                        .expect("actual destination reference"),
                )
            }));
            assert!(item.delivery_diagnostic().is_none());
            item.saved_sim_source(cx)
                .expect("export captured actual editable source")
                .clone()
        });
        let exact_source = derived
            .version()
            .editable_source
            .as_ref()
            .expect("actual captured pre-export editable source");
        assert_ne!(exact_source, &latest_saved);
        let read_container = |source: &sim_runtime::ProjectEditableSourceReference| {
            let bytes = fixture
                .assets
                .lock()
                .expect("actual editable source reader")
                .read_verified(
                    &source.identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
                )
                .expect("actual immutable editable source");
            assert_eq!(
                source.content_sha256,
                format!("{:x}", Sha256::digest(&bytes))
            );
            sim_runtime::SimDocument::decode(&bytes).expect("actual container integrity")
        };
        let before_export = read_container(exact_source);
        let after_export = read_container(&latest_saved);
        assert_eq!(before_export.documents, after_export.documents);
        assert!(!before_export.build_outputs.contains(&reference));
        assert!(after_export.build_outputs.contains(&reference));
        assert_eq!(
            derived.version().provenance["editable_source"],
            serde_json::to_value(exact_source).expect("actual source provenance")
        );
        assert_eq!(
            derived.version().provenance["delivery_profile"],
            serde_json::json!({"format":"png","color_space":"sRGB","alpha":"straight"})
        );
        assert_eq!(
            derived.version().provenance["output"]["pixels_sha256"],
            format!("{:x}", Sha256::digest(&rgba))
        );
        assert_eq!(
            derived.version().provenance["output"]["encoded_sha256"],
            exported.sha256
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual service")
                .read_verified(
                    &saved.identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64
                )
                .expect("first editable save remains immutable"),
            saved_bytes
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual version catalog")
                .resolve_project_asset_version(
                    &input_version,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("editable input remains pinned")
                .bytes(),
            original_bytes
        );
        let mut missing = container.clone();
        missing
            .documents
            .first_mut()
            .expect("actual saved engine")
            .dependencies[0]
            .asset_id = uuid::Uuid::new_v4();
        let proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            "missing-version",
            "sim",
            0,
            0,
            0,
            missing
                .encode()
                .expect("valid envelope with absent real dependency"),
        )
        .expect("actual negative proposal");
        let opaque_payload = "  \n  opaque other-engine fixture: 雪 / café / λ\n\tend  ";
        let opaque_engine = sim_runtime::SimEngineDocument::new(
            "c1ce41f9-0ad8-4b28-9601-810879559a20"
                .parse()
                .expect("stable opaque fixture ID"),
            "opaque-other-engine-fixture".into(),
            "qualified-fixture-pin-not-an-engine-admission".into(),
            "fixture.opaque-text.v1".into(),
            opaque_payload.into(),
            vec![input_version.clone()],
        );
        let mut mixed = after_export.clone();
        mixed.documents.push(opaque_engine.clone());
        mixed.provenance["opaque_fixture"] = serde_json::json!({"purpose":"nonexecuted engine preservation", "whitespace":"  \n\t", "unicode":"雪"});
        assert!(mixed.build_outputs.contains(&reference));
        let mixed_bytes = mixed
            .encode()
            .expect("valid real Graphite plus opaque-engine container");
        let mixed_proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            "mixed-engine-preservation",
            "sim",
            0,
            0,
            0,
            mixed_bytes.clone(),
        )
        .expect("actual opaque-preservation proposal");
        let mixed_identity = {
            let mut assets = fixture.assets.lock().expect("same real service");
            let mut committer = committer.lock().expect("same real publisher");
            let receipts = committer
                .commit_proposal_batch_and_register(
                    &[proposal, mixed_proposal],
                    "2026-10-10T12:00:00+00:00"
                        .parse()
                        .expect("fixed actual timestamp"),
                    &mut assets,
                    &writer,
                    &sim_tensor::CancellationToken::default(),
                )
                .expect("actual negative and mixed-container publication");
            assert_eq!(receipts.len(), 2);
            receipts
                .get(1)
                .expect("actual mixed-container committed receipt")
                .operation()
                .identity
                .clone()
        };
        let (before_revoke_revision, before_revoke_parameters, before_revoke_placement) = session
            .read_with(window, |session, _| {
                (
                    session.revision(),
                    session.parameters(),
                    session
                        .image_placement()
                        .expect("editable placement before grant retirement"),
                )
            });
        click_without_pumping(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        window.update(|_, cx| {
            crate::clear_asset_binding(cx)
                .expect("actual host grant retirement during pending export")
        });
        window.run_until_parked();
        paint(window);
        let after_revoke_outputs = output_records(&fixture, &reader, "/baked");
        assert_eq!(after_revoke_outputs.len(), 1);
        assert_eq!(
            after_revoke_outputs
                .first()
                .expect("only earlier committed PNG remains")
                .identity,
            exported.identity
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_some()
        );
        reopened.read_with(window, |item, _| {
            assert!(item.delivery_status().is_none());
            assert_eq!(
                item.delivery_diagnostic(),
                Some(
                    crate::session::ArtboardSessionError::ResourceRevoked
                        .to_string()
                        .as_str()
                )
            );
        });
        session.read_with(window, |session, _| {
            assert!(session.resource_is_revoked());
            assert_eq!(session.revision(), before_revoke_revision);
            assert_eq!(session.parameters(), before_revoke_parameters);
            assert_eq!(
                session
                    .image_placement()
                    .expect("editable revoked placement retained")
                    .to_cols_array()
                    .map(f64::to_bits),
                before_revoke_placement.to_cols_array().map(f64::to_bits)
            );
            assert!(!session.preview_is_current(&current));
        });
        drop(session);
        close_artboard(&workspace, reopened, window).await;
        window.update(|_, cx| {
            crate::register_asset_binding_with_output_reader(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                Some(reader.clone()),
                cx,
            )
            .expect("new authorized reader after actual revocation");
            crate::register_asset_publisher(committer.clone(), writer.clone(), cx)
                .expect("same canonical publisher reactivated");
        });
        let modal = saved_sim_search(&workspace, "missing-version", window);
        modal.read_with(window, |modal, _| {
            assert_eq!(modal.choices.len(), 1);
            assert!(modal.diagnostic.is_none());
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(window.debug_bounds("GRAPHITE-SIM-OPEN-ERROR").is_some());
        modal.read_with(window, |modal, _| {
            assert!(modal.diagnostic.as_ref().is_some_and(|message| {
                message.contains("immutable project asset version is missing")
            }))
        });
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        let mixed_modal = saved_sim_search(&workspace, "mixed-engine-preservation", window);
        mixed_modal.read_with(window, |modal, _| {
            assert_eq!(modal.choices.len(), 1);
            assert!(modal.diagnostic.is_none());
            assert_eq!(
                modal
                    .choices
                    .first()
                    .expect("actual published mixed .sim")
                    .label(),
                mixed_identity.relative_path.to_string_lossy()
            );
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        let mixed_item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("real mixed .sim adapter opener");
        let mixed_session = mixed_item.read_with(window, |item, _| item.session().clone());
        let (mixed_revision, mixed_parameters, mixed_source) =
            mixed_session.read_with(window, |session, _| {
                let preview = session
                    .preview()
                    .expect("original Graphite preview in mixed container");
                assert!(session.preview_is_current(preview));
                (
                    session.revision(),
                    session.parameters(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("same verified source")
                        .source()
                        .clone(),
                )
            });
        edit_artboard_field(&mixed_item, 9, "40", window);
        mixed_session.read_with(window, |session, _| {
            assert_eq!(session.revision(), mixed_revision + 1);
            assert_eq!(session.parameters(), mixed_parameters);
            assert_eq!(
                session
                    .image_placement()
                    .expect("only Graphite placement edited")
                    .translation
                    .x
                    .to_bits(),
                40_f64.to_bits()
            );
            assert!(std::sync::Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("immutable input unchanged")
                    .source(),
                &mixed_source
            ));
            assert!(session.preview_is_current(session.preview().expect("updated original scene")));
        });
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        mixed_item.read_with(window, |item, _| {
            assert!(item.delivery_diagnostic().is_none())
        });
        let mixed_saved = mixed_item.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("actual mixed-container save")
                .clone()
        });
        let mixed_saved_bytes = fixture
            .assets
            .lock()
            .expect("same real service")
            .read_verified(
                &mixed_saved.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("actual saved mixed-container bytes");
        assert_eq!(
            mixed_saved.content_sha256,
            format!("{:x}", Sha256::digest(&mixed_saved_bytes))
        );
        let mixed_after = sim_runtime::SimDocument::decode(&mixed_saved_bytes)
            .expect("actual mixed-container integrity");
        assert_eq!(mixed_after.documents.len(), mixed.documents.len());
        assert_eq!(
            mixed_after
                .documents
                .iter()
                .map(|document| document.document_id)
                .collect::<Vec<_>>(),
            mixed
                .documents
                .iter()
                .map(|document| document.document_id)
                .collect::<Vec<_>>()
        );
        let retained = mixed_after
            .documents
            .iter()
            .find(|document| document.document_id == opaque_engine.document_id)
            .expect("opaque engine row retained");
        assert_eq!(retained, &opaque_engine);
        assert_eq!(retained.payload, opaque_payload);
        assert_eq!(
            retained.payload_sha256,
            format!("{:x}", Sha256::digest(opaque_payload.as_bytes()))
        );
        assert_eq!(mixed_after.provenance, mixed.provenance);
        assert_eq!(mixed_after.build_outputs, mixed.build_outputs);
        assert_eq!(mixed_after.project_id, mixed.project_id);
        assert_eq!(mixed_after.format, mixed.format);
        assert_eq!(mixed_after.schema_version, mixed.schema_version);
        let graphite_before = mixed
            .documents
            .iter()
            .find(|document| document.engine == "graphite")
            .expect("verified original Graphite row");
        let graphite_after = mixed_after
            .documents
            .iter()
            .find(|document| document.engine == "graphite")
            .expect("only edited Graphite row");
        assert_eq!(graphite_after.document_id, graphite_before.document_id);
        assert_eq!(
            graphite_after.source_revision,
            graphite_before.source_revision
        );
        assert_eq!(
            graphite_after.serialization_format,
            graphite_before.serialization_format
        );
        assert_eq!(graphite_after.dependencies, graphite_before.dependencies);
        assert_ne!(graphite_after.payload, graphite_before.payload);
        assert_ne!(
            graphite_after.payload_sha256,
            graphite_before.payload_sha256
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual immutable mixed input")
                .read_verified(
                    &mixed_identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64
                )
                .expect("prior published mixed container unchanged"),
            mixed_bytes
        );
        drop(mixed_session);
        close_artboard(&workspace, mixed_item, window).await;
    }

    #[gpui::test]
    async fn at_task573_invalid_native_placement_blocks_real_save_and_export_until_escape_retry(
        cx: &mut TestAppContext,
    ) {
        use sha2::{Digest as _, Sha256};
        let fixture = host_reader_fixture().expect("actual authorized asset service");
        let png = indexed_request_png(&fixture, 4).expect("actual PNG input");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, _) = register_fixture_workflow(&fixture, &png, window);
        let item = open_requested_png(&workspace, window);
        let session = item.read_with(window, |item, _| item.session().clone());
        let input_version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("genuine immutable input")
                .clone()
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let prepared = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("current canonical PNG before local draft")
                .clone()
        });
        let (revision, parameters, placement, network, source, preview) =
            session.read_with(window, |session, _| {
                let preview = session.preview().expect("real current initial preview");
                assert!(session.preview_is_current(preview));
                assert!(session.png_is_current(&prepared));
                (
                    session.revision(),
                    session.parameters(),
                    session.image_placement().expect("encoded image placement"),
                    session.document().network().clone(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("exact retained input")
                        .source()
                        .clone(),
                    preview.clone(),
                )
            });
        let presentation = item.read_with(window, |item, _| item.presentation().clone());
        let editor = item.read_with(window, |item, _| {
            item.artboard_editors
                .get(9)
                .expect("native image X field")
                .clone()
        });
        let canonical_text = editor.read_with(window, |editor, cx| editor.text(cx));
        let saved_documents = || {
            output_records(&fixture, &reader, ".sim")
                .into_iter()
                .filter(|record| {
                    record
                        .identity
                        .relative_path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        == Some("sim")
                })
                .collect::<Vec<_>>()
        };
        let baseline_outputs = output_records(&fixture, &reader, "");
        let baseline_pins = output_records(&fixture, &reader, ".sim/inputs/");
        assert_eq!(baseline_pins.len(), 1);
        let baseline_operations = committer
            .lock()
            .expect("actual canonical publication journal")
            .operations()
            .to_vec();
        edit_artboard_field(&item, 9, "not-a-number", window);
        assert_eq!(
            editor.read_with(window, |editor, cx| editor.text(cx)),
            "not-a-number"
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        assert!(saved_documents().is_empty());
        assert!(output_records(&fixture, &reader, "/baked").is_empty());
        for selector in ["GRAPHITE-ARTBOARD-SAVE-SIM", "GRAPHITE-ARTBOARD-EXPORT-PNG"] {
            click(window, selector);
            assert!(
                window
                    .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                    .is_some()
            );
            item.read_with(window, |item, cx| {
                assert_eq!(
                    item.delivery_diagnostic(),
                    Some(
                        "Apply or discard the current composition edits before saving or exporting."
                    )
                );
                assert!(item.delivery_status().is_none());
                assert!(item.saved_sim_source(cx).is_none());
                assert_eq!(item.input_version(cx), Some(&input_version));
                assert_eq!(
                    item.prepared_png()
                        .expect("canonical prepared PNG retained")
                        .pixels_sha256(),
                    prepared.pixels_sha256()
                );
            });
            assert_eq!(
                editor.read_with(window, |editor, cx| editor.text(cx)),
                "not-a-number"
            );
            assert!(saved_documents().is_empty());
            assert!(output_records(&fixture, &reader, "/baked").is_empty());
            assert_eq!(output_records(&fixture, &reader, ""), baseline_outputs);
            assert_eq!(
                output_records(&fixture, &reader, ".sim/inputs/"),
                baseline_pins
            );
            assert_eq!(
                committer
                    .lock()
                    .expect("invalid draft created no canonical operation")
                    .operations(),
                baseline_operations
            );
            session.read_with(window, |session, _| {
                assert_eq!(session.revision(), revision);
                assert_eq!(session.parameters(), parameters);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("committed placement preserved")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert_eq!(session.document().network(), &network);
                assert!(std::sync::Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("same immutable source")
                        .source(),
                    &source
                ));
                assert!(session.preview_is_current(&preview));
                assert!(session.png_is_current(&prepared));
                assert!(std::sync::Arc::ptr_eq(
                    session.preview().expect("same canonical frame").image(),
                    preview.image()
                ));
            });
            presentation.read_with(window, |presentation, _| {
                assert!(std::sync::Arc::ptr_eq(
                    presentation.image().expect("last valid displayed frame"),
                    preview.image()
                ))
            });
        }
        editor.update_in(window, |editor, window, cx| {
            editor.focus_handle(cx).focus(window, cx)
        });
        window.simulate_keystrokes("escape");
        window.run_until_parked();
        paint(window);
        assert_eq!(
            editor.read_with(window, |editor, cx| editor.text(cx)),
            canonical_text
        );
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision);
            assert!(session.preview_is_current(&preview));
            assert!(session.png_is_current(&prepared));
        });
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_none()
        );
        assert_eq!(saved_documents().len(), 1);
        let saved = item.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            item.saved_sim_source(cx)
                .expect("actual retry saved editable source")
                .clone()
        });
        let saved_bytes = fixture
            .assets
            .lock()
            .expect("real service")
            .read_verified(
                &saved.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("actual immutable saved .sim");
        assert_eq!(
            saved.content_sha256,
            format!("{:x}", Sha256::digest(&saved_bytes))
        );
        let container = sim_runtime::SimDocument::decode(&saved_bytes)
            .expect("actual saved container integrity");
        assert_eq!(
            container
                .documents
                .first()
                .expect("original Graphite source")
                .dependencies,
            vec![input_version]
        );
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-RESULT")
                .is_some()
        );
        let outputs = output_records(&fixture, &reader, "/baked");
        assert_eq!(outputs.len(), 1);
        let output = outputs.first().expect("actual retry delivered PNG");
        let bytes = fixture
            .assets
            .lock()
            .expect("real service")
            .read_verified(
                &output.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("actual published retry PNG");
        assert_eq!(bytes, prepared.png().encoded_bytes());
        assert_eq!(output.sha256, format!("{:x}", Sha256::digest(&bytes)));
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision);
            assert_eq!(session.document().network(), &network);
            assert!(std::sync::Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("editable input retained after delivery")
                    .source(),
                &source
            ));
            assert!(session.preview_is_current(&preview));
        });
        drop(editor);
        drop(presentation);
        drop(session);
        close_artboard(&workspace, item, window).await;
    }

    #[gpui::test]
    async fn at_task573_linked_open_revocation_and_publisher_replacement_before_read_cannot_pin_or_install(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real shared service");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed Input PNG");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, writer) = register_fixture_workflow(&fixture, &png, window);
        let before = committer
            .lock()
            .expect("actual publisher journal")
            .operations()
            .to_vec();
        assert!(output_records(&fixture, &reader, ".sim/inputs/").is_empty());
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let modal = chooser(&workspace, window);
        search_requested(&modal, window);
        click_without_pumping(window, "GRAPHITE-PNG-ASSET-0");
        let cancellation = modal.read_with(window, |modal, _| {
            assert!(modal.opening);
            modal
                .open_cancellation
                .as_ref()
                .expect("tracked linked opener authority")
                .clone()
        });
        window.update(|_, cx| {
            crate::register_asset_publisher(committer.clone(), writer.clone(), cx)
                .expect("exact duplicate publisher registration");
        });
        assert!(!cancellation.is_cancelled());
        assert_eq!(
            committer
                .lock()
                .expect("no scheduled publication")
                .operations(),
            before
        );
        window.update(|_, cx| {
            crate::clear_asset_binding(cx).expect("actual revocation before background read")
        });
        assert!(cancellation.is_cancelled());
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_some());
        modal.read_with(window, |modal, _| {
            assert!(!modal.opening);
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        assert_eq!(
            committer
                .lock()
                .expect("actual publisher after revoked request")
                .operations(),
            before
        );
        assert!(output_records(&fixture, &reader, ".sim/inputs/").is_empty());
        click(window, "GRAPHITE-PNG-CANCEL");
        window.update(|_, cx| {
            crate::register_asset_binding_with_output_reader(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                Some(reader.clone()),
                cx,
            )
            .expect("new reader after revocation");
            crate::register_asset_publisher(committer.clone(), writer.clone(), cx)
                .expect("same canonical publisher");
        });
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let replacement_modal = chooser(&workspace, window);
        search_requested(&replacement_modal, window);
        let binding =
            window.update(|_, cx| crate::asset_binding(cx).expect("current actual binding"));
        click_without_pumping(window, "GRAPHITE-PNG-ASSET-0");
        let replacement_cancellation = replacement_modal.read_with(window, |modal, _| {
            modal
                .open_cancellation
                .as_ref()
                .expect("tracked same-binding opener")
                .clone()
        });
        let roots = fixture
            .assets
            .lock()
            .expect("actual shared roots")
            .roots()
            .clone();
        let replacement = std::sync::Arc::new(std::sync::Mutex::new(
            sim_runtime::OutputCommitter::open(roots).expect("actual replacement publisher"),
        ));
        assert!(!std::sync::Arc::ptr_eq(&replacement, &committer));
        window.update(|_, cx| {
            crate::register_asset_publisher(replacement.clone(), writer.clone(), cx)
                .expect("same binding publisher replacement")
        });
        assert!(replacement_cancellation.is_cancelled());
        window.update(|_, cx| {
            assert_eq!(
                crate::asset_binding(cx)
                    .expect("reader identity unchanged")
                    .entity_id(),
                binding.entity_id()
            )
        });
        window.run_until_parked();
        paint(window);
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(window.debug_bounds("GRAPHITE-PNG-OPEN-ERROR").is_some());
        assert_eq!(
            replacement
                .lock()
                .expect("replacement journal unchanged")
                .operations(),
            before
        );
        assert_eq!(
            committer
                .lock()
                .expect("old journal unchanged")
                .operations(),
            before
        );
        assert!(output_records(&fixture, &reader, ".sim/inputs/").is_empty());
        click(window, "GRAPHITE-PNG-CANCEL");
        window.dispatch_action(OpenPngEditor);
        window.run_until_parked();
        paint(window);
        let retry = chooser(&workspace, window);
        search_requested(&retry, window);
        click_without_pumping(window, "GRAPHITE-PNG-ASSET-0");
        let retry_cancellation = retry.read_with(window, |modal, _| {
            modal
                .open_cancellation
                .as_ref()
                .expect("new genuine retry authority")
                .clone()
        });
        window.update(|_, cx| {
            crate::register_asset_publisher(replacement.clone(), writer.clone(), cx)
                .expect("duplicate current publisher preserves operation")
        });
        assert!(!retry_cancellation.is_cancelled());
        window.run_until_parked();
        paint(window);
        let item = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("authorized retry genuinely installed Item");
        assert_eq!(output_records(&fixture, &reader, ".sim/inputs/").len(), 1);
        assert_eq!(
            replacement
                .lock()
                .expect("exactly one retry pin")
                .operations()
                .len(),
            before.len() + 1
        );
        item.read_with(window, |item, cx| {
            let session = item.session().read(cx);
            assert!(
                session
                    .preview_is_current(session.preview().expect("actual automatic retry preview"))
            );
        });
        close_artboard(&workspace, item, window).await;
    }

    #[gpui::test]
    async fn at_task573_update_input_revoked_before_read_preserves_pin_and_editable_state(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("actual shared service");
        let png = indexed_request_png(&fixture, 4).expect("actual original PNG");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, writer) = register_fixture_workflow(&fixture, &png, window);
        let item = open_requested_png(&workspace, window);
        edit_artboard_field(&item, 9, "12", window);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("actual pinned input before update")
                .clone()
        });
        let png_receipt = item.read_with(window, |item, _| {
            item.prepared_png()
                .expect("genuine prepared edited PNG")
                .clone()
        });
        let session = item.read_with(window, |item, _| item.session().clone());
        let (revision, parameters, placement, network, source, preview) =
            session.read_with(window, |session, _| {
                let preview = session.preview().expect("actual edited native frame");
                assert!(session.preview_is_current(preview));
                assert!(session.png_is_current(&png_receipt));
                (
                    session.revision(),
                    session.parameters(),
                    session.image_placement().expect("current native placement"),
                    session.document().network().clone(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("retained initial seal")
                        .source()
                        .clone(),
                    preview.clone(),
                )
            });
        let changed = indexed_request_png(&fixture, 3).expect("genuine changed producer bytes");
        assert_ne!(changed.encoded, png.encoded);
        let before = committer
            .lock()
            .expect("actual journal before update")
            .operations()
            .to_vec();
        assert_eq!(output_records(&fixture, &reader, ".sim/inputs/").len(), 1);
        click_without_pumping(window, "GRAPHITE-ARTBOARD-UPDATE-INPUT");
        item.read_with(window, |item, _| {
            assert_eq!(
                item.delivery_status(),
                Some("Checking the original PNG for a new version")
            )
        });
        window.update(|_, cx| {
            crate::register_asset_publisher(committer.clone(), writer, cx)
                .expect("duplicate publisher does not supersede update")
        });
        assert_eq!(
            committer
                .lock()
                .expect("background update not yet executed")
                .operations(),
            before
        );
        window.update(|_, cx| {
            crate::clear_asset_binding(cx)
                .expect("actual update revocation before first background read")
        });
        window.run_until_parked();
        paint(window);
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-DELIVERY-ERROR")
                .is_some()
        );
        item.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&version));
            assert!(item.delivery_status().is_none());
            assert!(
                item.delivery_diagnostic()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        assert_eq!(
            committer
                .lock()
                .expect("revoked update published no new pin")
                .operations(),
            before
        );
        let pins = output_records(&fixture, &reader, ".sim/inputs/");
        assert_eq!(pins.len(), 1);
        let retained = fixture
            .assets
            .lock()
            .expect("actual project versions")
            .resolve_project_asset_version(
                &version,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("existing immutable input version survives revocation");
        assert_eq!(retained.bytes(), png.encoded);
        session.read_with(window, |session, _| {
            assert!(session.resource_is_revoked());
            assert_eq!(session.revision(), revision);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(
                session
                    .image_placement()
                    .expect("editable placement remains")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(session.document().network(), &network);
            assert!(std::sync::Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("original source retained")
                    .source(),
                &source
            ));
            assert!(!session.preview_is_current(&preview));
            assert!(!session.png_is_current(&png_receipt));
        });
        assert_eq!(
            workspace.read_with(window, |workspace, cx| workspace
                .active_item_as::<GraphiteArtboardItem>(cx)
                .expect("same editable Item remains")
                .entity_id()),
            item.entity_id()
        );
        drop(session);
        close_artboard(&workspace, item, window).await;
    }
    #[gpui::test]
    async fn at_task573_split_input_update_save_and_export_share_current_links_and_retained_outputs(
        cx: &mut TestAppContext,
    ) {
        use sha2::{Digest as _, Sha256};

        let fixture = host_reader_fixture().expect("real shared project service");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed Input PNG");
        let original_bytes = png.encoded.clone();
        let (workspace, window) = native_workspace(cx).await;
        let (reader, _, _) = register_fixture_workflow(&fixture, &png, window);
        let first = open_requested_png(&workspace, window);
        edit_artboard_field(&first, 5, "16", window);
        edit_artboard_field(&first, 8, "8", window);
        edit_artboard_field(&first, 9, "12", window);
        edit_artboard_field(&first, 10, "8", window);
        let session = first.read_with(window, |item, _| item.session().clone());
        let original_version = first.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("actual pinned version")
                .clone()
        });
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        first.read_with(window, |item, _| {
            assert!(item.delivery_diagnostic().is_none());
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Exported baked PNG:"))
            );
        });
        let first_saved = first.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("actual final saved container")
                .clone()
        });
        let read_container = |source: &sim_runtime::ProjectEditableSourceReference| {
            let bytes = fixture
                .assets
                .lock()
                .expect("actual shared reader")
                .read_verified(
                    &source.identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
                )
                .expect("actual saved container bytes");
            assert_eq!(
                source.content_sha256,
                format!("{:x}", Sha256::digest(&bytes))
            );
            sim_runtime::SimDocument::decode(&bytes).expect("actual container integrity")
        };
        let first_container = read_container(&first_saved);
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let old_png = first.read_with(window, |item, _| {
            item.prepared_png()
                .expect("genuine prepared PNG before split update")
                .clone()
        });
        assert!(session.read_with(window, |session, _| session.png_is_current(&old_png)));
        assert_eq!(first_container.build_outputs.len(), 1);
        let first_output = first_container
            .build_outputs
            .first()
            .expect("genuine initial published output")
            .clone();
        let (parameters, configuration, placement, revision, old_preview, old_source) = session
            .read_with(window, |session, _| {
                let preview = session.preview().expect("original visible preview");
                assert!(session.preview_is_current(preview));
                (
                    session.parameters(),
                    session.configuration(),
                    session
                        .image_placement()
                        .expect("original edited placement"),
                    session.revision(),
                    preview.clone(),
                    session
                        .document()
                        .encoded_png_input()
                        .expect("original encoded owner")
                        .source()
                        .clone(),
                )
            });
        let first_pane =
            workspace.read_with(window, |workspace, _| workspace.active_pane().clone());
        let split_pane = workspace
            .update_in(window, |workspace, window, cx| {
                workspace.split_and_clone(
                    first_pane.clone(),
                    workspace::SplitDirection::Right,
                    window,
                    cx,
                )
            })
            .await
            .expect("actual Workspace split-and-clone route");
        let second = split_pane
            .read_with(window, |pane, _cx| {
                pane.active_item()
                    .and_then(|item| item.downcast::<GraphiteArtboardItem>())
            })
            .expect("native Artboard in the split pane");
        window.run_until_parked();
        paint(window);
        assert_ne!(first.entity_id(), second.entity_id());
        second.read_with(window, |item, cx| {
            assert_eq!(item.session(), &session);
            assert_eq!(item.presentation(), first.read(cx).presentation());
            assert_eq!(item.input_version(cx), Some(&original_version));
            assert_eq!(item.saved_sim_source(cx), Some(&first_saved));
        });

        let changed =
            indexed_request_png(&fixture, 3).expect("actual originating Input replacement");
        assert_eq!(changed.asset_identity, png.asset_identity);
        assert_ne!(changed.encoded, original_bytes);
        let update = first
            .update_in(window, |item, window, cx| {
                item.update_input_version(window, cx)
            })
            .expect("existing explicit Update input API on the first split");
        assert!(update.await.expect("genuine shared source update"));
        window.run_until_parked();
        paint(window);
        let updated_version = first.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            item.input_version(cx)
                .expect("new linked immutable version")
                .clone()
        });
        assert_eq!(updated_version.asset_id, original_version.asset_id);
        assert_ne!(updated_version.version_id, original_version.version_id);
        assert_eq!(
            updated_version.content_sha256,
            format!("{:x}", Sha256::digest(&changed.encoded))
        );
        second.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&updated_version));
            assert_eq!(
                item.saved_sim_source(cx),
                first.read(cx).saved_sim_source(cx)
            );
            assert!(!item.session().read(cx).preview_is_current(&old_preview));
            assert!(!item.session().read(cx).png_is_current(&old_png));
        });
        let (updated_source, expected_frame, network, document_encoding, node_id) = session
            .read_with(window, |session, _| {
                assert_eq!(session.revision(), revision + 1);
                assert_eq!(session.parameters(), parameters);
                assert_eq!(session.configuration(), configuration);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("placement survives dependency update")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert!(!session.preview_is_current(&old_preview));
                let preview = session
                    .preview()
                    .expect("automatic current updated preview");
                assert!(session.preview_is_current(preview));
                let frame = preview
                    .image()
                    .as_bytes(0)
                    .expect("actual native BGRA frame");
                assert_ne!(
                    frame,
                    old_preview
                        .image()
                        .as_bytes(0)
                        .expect("retained historical frame")
                );
                let (pixels, remainder) = frame.as_chunks::<4>();
                assert!(remainder.is_empty());
                assert!(
                    pixels
                        .iter()
                        .any(|pixel| pixel[3] == 255 && pixel[..3] != [255, 255, 255])
                );
                let source = session
                    .document()
                    .encoded_png_input()
                    .expect("current encoded checkpoint")
                    .source()
                    .clone();
                assert!(!std::sync::Arc::ptr_eq(&source, &old_source));
                assert_eq!(
                    source.identity().version_id.as_ref(),
                    updated_version.version_id.to_string()
                );
                assert_eq!(
                    source.identity().content_sha256,
                    changed.image_identity.content_sha256
                );
                (
                    source,
                    frame.to_vec(),
                    session.document().network().clone(),
                    session.document().document_id_encoding(),
                    session.document().node_id(),
                )
            });
        let saved_receipt = second
            .update(window, |item, cx| item.save_sim(cx))
            .expect("actual Save from the other live split");
        paint(window);
        let saved_from_second = second.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert_eq!(item.input_version(cx), Some(&updated_version));
            item.saved_sim_source(cx)
                .expect("save from the surviving split")
                .clone()
        });
        assert_eq!(
            saved_receipt.operation().identity,
            saved_from_second.identity
        );
        assert_eq!(
            first.read_with(window, |item, cx| item.saved_sim_source(cx).cloned()),
            Some(saved_from_second.clone())
        );
        let second_container = read_container(&saved_from_second);
        assert_eq!(second_container.documents.len(), 1);
        assert_eq!(
            second_container
                .documents
                .first()
                .expect("same Graphite engine")
                .document_id,
            first_container
                .documents
                .first()
                .expect("original engine")
                .document_id
        );
        assert_eq!(
            second_container
                .documents
                .first()
                .expect("updated payload dependency")
                .dependencies,
            vec![updated_version.clone()]
        );
        assert_eq!(
            second_container.build_outputs,
            first_container.build_outputs
        );
        let mut rgba = expected_frame.clone();
        let (pixels, remainder) = rgba.as_chunks_mut::<4>();
        assert!(remainder.is_empty());
        for pixel in pixels {
            pixel.swap(0, 2);
        }
        let export = second
            .update(window, |item, cx| item.export_baked_png(cx))
            .expect("actual Export from the other live split");
        let export_receipt = export
            .await
            .expect("genuine baked publication from the shared current source");
        window.run_until_parked();
        paint(window);
        let final_saved = second.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Exported baked PNG:"))
            );
            item.saved_sim_source(cx)
                .expect("final shared container with both outputs")
                .clone()
        });
        assert_eq!(
            first.read_with(window, |item, cx| item.saved_sim_source(cx).cloned()),
            Some(final_saved.clone())
        );
        let final_container = read_container(&final_saved);
        assert_eq!(final_container.documents, second_container.documents);
        assert_eq!(final_container.build_outputs.len(), 2);
        assert!(final_container.build_outputs.contains(&first_output));
        let derived_reference = final_container
            .build_outputs
            .iter()
            .find(|reference| *reference != &first_output)
            .expect("genuine later output reference")
            .clone();
        let derived = fixture
            .assets
            .lock()
            .expect("actual retained version catalog")
            .resolve_project_asset_version(
                &derived_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("actual baked output with authenticated historical dependency links");
        assert_eq!(
            derived.version().stored_identity,
            export_receipt.operation().identity
        );
        assert_eq!(
            derived.version().dependencies,
            vec![updated_version.clone()]
        );
        assert_eq!(
            derived.version().editable_source,
            Some(saved_from_second.clone())
        );
        assert_eq!(
            derived.version().provenance["input_versions"],
            serde_json::json!([updated_version])
        );
        assert_eq!(
            derived.version().provenance["editable_source"],
            serde_json::to_value(&saved_from_second).expect("exact captured source reference")
        );
        assert_eq!(
            derived.version().provenance["delivery_profile"],
            serde_json::json!({"format":"png","color_space":"sRGB","alpha":"straight"})
        );
        assert_eq!(
            derived_reference.content_sha256,
            format!("{:x}", Sha256::digest(derived.bytes()))
        );
        let decoded = image::load_from_memory_with_format(derived.bytes(), image::ImageFormat::Png)
            .expect("actual final PNG")
            .into_rgba8();
        assert_eq!(decoded.dimensions(), (1920_u32, 1080_u32));
        assert_eq!(decoded.as_raw(), &rgba);
        assert_eq!(
            derived.version().provenance["output"]["pixels_sha256"],
            format!("{:x}", Sha256::digest(&rgba))
        );
        assert_eq!(output_records(&fixture, &reader, "/baked").len(), 2);
        assert_eq!(read_container(&first_saved), first_container);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual immutable old version")
                .resolve_project_asset_version(
                    &original_version,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024,
                )
                .expect("original pinned bytes survive split update")
                .bytes(),
            original_bytes
        );
        let first_save_after_export = first
            .update(window, |item, cx| item.save_sim(cx))
            .expect("actual Save in the first split after the other split exported");
        paint(window);
        let latest_saved = first.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("shared latest saved authority")
                .clone()
        });
        assert_eq!(
            latest_saved.identity,
            first_save_after_export.operation().identity
        );
        assert_ne!(latest_saved, final_saved);
        second.read_with(window, |item, cx| {
            assert_eq!(item.saved_sim_source(cx), Some(&latest_saved));
            assert_eq!(item.input_version(cx), Some(&updated_version));
        });
        let latest_container = read_container(&latest_saved);
        assert_eq!(latest_container.documents, final_container.documents);
        assert_eq!(
            latest_container.build_outputs,
            final_container.build_outputs
        );
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&first, true, true, window, cx));
        });
        close_artboard(&workspace, first, window).await;
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&second, true, true, window, cx));
        });
        close_artboard(&workspace, second, window).await;
        let modal = saved_sim_search(
            &workspace,
            &latest_saved.identity.relative_path.to_string_lossy(),
            window,
        );
        modal.read_with(window, |modal, _| {
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("actual saved opener after both original views closed");
        reopened.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&updated_version));
            assert_eq!(item.saved_sim_source(cx), Some(&latest_saved));
            assert_ne!(item.session(), &session);
            let restored = item.session().read(cx);
            assert_eq!(restored.revision(), revision + 1);
            assert_eq!(restored.parameters(), parameters);
            assert_eq!(restored.configuration(), configuration);
            assert_eq!(
                restored
                    .image_placement()
                    .expect("restored exact placement")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(
                restored.document().document_id_encoding(),
                document_encoding
            );
            assert_eq!(restored.document().node_id(), node_id);
            assert_eq!(restored.document().network(), &network);
            let source = restored
                .document()
                .encoded_png_input()
                .expect("restored immutable source")
                .source();
            assert_eq!(source.identity(), updated_source.identity());
            assert!(!std::sync::Arc::ptr_eq(source, &updated_source));
            let preview = restored
                .preview()
                .expect("automatic restored visible result");
            assert!(restored.preview_is_current(preview));
            assert_eq!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("exact restored native frame"),
                expected_frame
            );
        });
        close_artboard(&workspace, reopened, window).await;
    }

    fn published_png_search(
        workspace: &Entity<Workspace>,
        text: &str,
        window: &mut VisualTestContext,
    ) -> Entity<AuthorizedPngChooser> {
        window.dispatch_action(OpenPublishedPng);
        window.run_until_parked();
        paint(window);
        let modal = chooser(workspace, window);
        let search = modal.read_with(window, |modal, _| modal.search_editor.clone());
        search.update_in(window, |editor, window, cx| {
            editor.set_text("", window, cx);
            editor.focus_handle(cx).focus(window, cx);
        });
        window.simulate_input(text);
        window.run_until_parked();
        paint(window);
        modal.read_with(window, |modal, _| {
            assert!(matches!(modal.kind, ChooserKind::PublishedPng));
            assert!(!modal.loading);
            assert!(!modal.opening);
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
            assert!(matches!(
                modal.choices.first(),
                Some(AssetChoice::PublishedPng(_))
            ));
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-PUBLISHED-PNG-CHOOSER")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-PUBLISHED-PNG-SEARCH")
                .is_some()
        );
        assert!(window.debug_bounds("GRAPHITE-PNG-CHOOSER").is_none());
        assert!(window.debug_bounds("GRAPHITE-SIM-CHOOSER").is_none());
        modal
    }

    #[gpui::test]
    async fn at_r28_published_png_native_link_update_save_reopen_and_derived_export_preserve_producer(
        cx: &mut TestAppContext,
    ) {
        use sha2::{Digest as _, Sha256};
        let fixture = host_reader_fixture().expect("real canonical asset service");
        let png = indexed_request_png(&fixture, 4).expect("actual original Input PNG");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, _) = register_fixture_workflow(&fixture, &png, window);
        let focus = workspace.read_with(window, |workspace, cx| workspace.focus_handle(cx));
        window.update(|window, cx| focus.focus(window, cx));
        paint(window);
        window.update(|window, cx| {
            assert!(window.available_actions(cx).iter().any(|action| {
                action.name() == <OpenPublishedPng as gpui::Action>::name_for_type()
            }));
        });
        let producer = open_requested_png(&workspace, window);
        let producer_input = producer.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("genuine producer input version")
                .clone()
        });
        edit_artboard_field(&producer, 5, "64", window);
        edit_artboard_field(&producer, 8, "32", window);
        edit_artboard_field(&producer, 9, "4", window);
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        let first_record = output_records(&fixture, &reader, "/baked")
            .into_iter()
            .next()
            .expect("actual first published PNG");
        let first_saved = producer.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            item.saved_sim_source(cx)
                .expect("producer final saved source")
                .clone()
        });
        let read_container = |source: &sim_runtime::ProjectEditableSourceReference| {
            let bytes = fixture
                .assets
                .lock()
                .expect("actual shared reader")
                .read_verified(
                    &source.identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
                )
                .expect("authentic saved source bytes");
            assert_eq!(
                source.content_sha256,
                format!("{:x}", Sha256::digest(&bytes))
            );
            (
                sim_runtime::SimDocument::decode(&bytes).expect("actual saved container"),
                bytes,
            )
        };
        let (first_container, first_saved_bytes) = read_container(&first_saved);
        let first_reference = first_container
            .build_outputs
            .first()
            .expect("actual published version reference")
            .clone();
        let first_output = fixture
            .assets
            .lock()
            .expect("canonical immutable catalog")
            .resolve_project_asset_version(
                &first_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("authentic committed producer with complete dependency/source closure");
        assert_eq!(
            first_output.version().stored_identity,
            first_record.identity
        );
        assert_eq!(
            first_output.version().dependencies,
            vec![producer_input.clone()]
        );
        assert_eq!(
            first_output.version().reference.content_sha256,
            first_record.sha256
        );
        let producer_source = first_output
            .version()
            .editable_source
            .as_ref()
            .expect("producer captured pre-export source")
            .clone();
        let (_, producer_source_bytes) = read_container(&producer_source);
        let producer_version = first_output.version().clone();
        let producer_bytes = first_output.bytes().to_vec();
        let producer_network = producer.read_with(window, |item, cx| {
            item.session().read(cx).document().network().clone()
        });
        let catalog_before_open = fixture
            .assets
            .lock()
            .expect("canonical catalog")
            .project_png_versions(&reader)
            .expect("actual registered versions");
        let operations_before_open = committer
            .lock()
            .expect("canonical journal")
            .operations()
            .to_vec();
        let modal = published_png_search(
            &workspace,
            &first_record.identity.relative_path.to_string_lossy(),
            window,
        );
        modal.read_with(window, |modal, _| {
            let selected = modal
                .choices
                .first()
                .expect("actual immutable Output choice");
            assert_eq!(selected.byte_size(), producer_bytes.len() as u64);
            assert!(
                selected
                    .label()
                    .contains(&first_reference.version_id.to_string())
            );
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        let consumer = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("native published-version opener installed editable consumer");
        assert_ne!(consumer, producer);
        let session = consumer.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&first_reference));
            assert!(item.saved_sim_source(cx).is_none());
            assert!(item.delivery_diagnostic().is_none());
            item.session().clone()
        });
        let old_preview = session.read_with(window, |session, _| {
            let source = session
                .document()
                .encoded_png_input()
                .expect("linked sealed original bytes")
                .source();
            assert_eq!(
                source.identity().version_id.as_ref(),
                first_reference.version_id.to_string()
            );
            assert_eq!(
                source.identity().content_sha256,
                <[u8; 32]>::from(Sha256::digest(&producer_bytes))
            );
            assert_eq!((source.width(), source.height()), (1920_u32, 1080_u32));
            let preview = session
                .preview()
                .expect("actual automatic published-image scene");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                (
                    preview.image().size(0).width.0,
                    preview.image().size(0).height.0
                ),
                (1920_i32, 1080_i32)
            );
            let (pixels, remainder) = preview
                .image()
                .as_bytes(0)
                .expect("native visible frame")
                .as_chunks::<4>();
            assert!(remainder.is_empty());
            assert!(pixels.iter().all(|pixel| pixel[3] == 255));
            preview.clone()
        });
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("same canonical catalog")
                .project_png_versions(&reader)
                .expect("registered versions after open"),
            catalog_before_open
        );
        assert_eq!(
            committer
                .lock()
                .expect("same canonical journal")
                .operations(),
            operations_before_open
        );
        edit_artboard_field(&consumer, 5, "1920", window);
        edit_artboard_field(&consumer, 8, "1080", window);
        edit_artboard_field(&consumer, 9, "24", window);
        let producer_rgba =
            image::load_from_memory_with_format(&producer_bytes, image::ImageFormat::Png)
                .expect("actual opaque producer PNG texel oracle")
                .into_rgba8();
        let expected_pixel = producer_rgba.get_pixel(52, 24).0;
        assert_eq!(expected_pixel[3], 255);
        assert_ne!(expected_pixel[..3], [255, 255, 255]);
        session.read_with(window, |session, _| {
            let preview = session
                .preview()
                .expect("explicitly placed source visible at native size");
            assert!(session.preview_is_current(preview));
            let (pixels, remainder) = preview
                .image()
                .as_bytes(0)
                .expect("actual visible native frame")
                .as_chunks::<4>();
            assert!(remainder.is_empty());
            assert_eq!(
                pixels.get(24 * 1920 + 76).copied(),
                Some([
                    expected_pixel[2],
                    expected_pixel[1],
                    expected_pixel[0],
                    expected_pixel[3]
                ])
            );
        });
        click(window, "GRAPHITE-ARTBOARD-PREPARE-PNG");
        let old_png = consumer.read_with(window, |item, _| {
            item.prepared_png()
                .expect("actual consumer PNG before explicit source update")
                .clone()
        });
        let (parameters, configuration, placement, revision) =
            session.read_with(window, |session, _| {
                assert!(!session.preview_is_current(&old_preview));
                assert!(session.png_is_current(&old_png));
                (
                    session.parameters(),
                    session.configuration(),
                    session.image_placement().expect("editable placement"),
                    session.revision(),
                )
            });
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&producer, true, true, window, cx));
        });
        edit_artboard_field(&producer, 9, "12", window);
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        let producer_latest_saved = producer.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("producer saved after actual second export")
                .clone()
        });
        let (producer_latest_container, producer_latest_bytes) =
            read_container(&producer_latest_saved);
        let second_reference = producer_latest_container
            .build_outputs
            .iter()
            .find(|reference| *reference != &first_reference)
            .expect("actual same-asset successor")
            .clone();
        assert_eq!(second_reference.asset_id, first_reference.asset_id);
        assert_ne!(second_reference.version_id, first_reference.version_id);
        let second_output = fixture
            .assets
            .lock()
            .expect("canonical producer successor")
            .resolve_project_asset_version(
                &second_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("actual successor complete closure");
        assert_ne!(second_output.bytes(), producer_bytes);
        let producer_current_checkpoint = producer.read_with(window, |item, cx| {
            let session = item.session().read(cx);
            (
                session.revision(),
                session.parameters(),
                session.configuration(),
                session.document().network().clone(),
            )
        });
        let second_version = second_output.version().clone();
        let second_bytes = second_output.bytes().to_vec();
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision);
            assert!(session.png_is_current(&old_png));
        });
        consumer.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&first_reference))
        });
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&consumer, true, true, window, cx));
        });
        let versions_before_update = fixture
            .assets
            .lock()
            .expect("real canonical catalog")
            .project_png_versions(&reader)
            .expect("registered versions before explicit update");
        let journal_before_update = committer
            .lock()
            .expect("real canonical journal")
            .operations()
            .to_vec();
        click(window, "GRAPHITE-ARTBOARD-UPDATE-INPUT");
        let (updated_network, updated_frame) = session.read_with(window, |session, _| {
            assert_eq!(session.revision(), revision + 1);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            assert_eq!(
                session
                    .image_placement()
                    .expect("retained placement")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert!(!session.png_is_current(&old_png));
            let source = session
                .document()
                .encoded_png_input()
                .expect("actual updated encoded checkpoint")
                .source();
            assert_eq!(
                source.identity().version_id.as_ref(),
                second_reference.version_id.to_string()
            );
            let preview = session.preview().expect("actual refreshed successor");
            assert!(session.preview_is_current(preview));
            (
                session.document().network().clone(),
                preview.image().as_bytes(0).expect("actual frame").to_vec(),
            )
        });
        consumer.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&second_reference));
            assert!(item.delivery_diagnostic().is_none());
        });
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("canonical catalog")
                .project_png_versions(&reader)
                .expect("no output repin"),
            versions_before_update
        );
        assert_eq!(
            committer.lock().expect("canonical journal").operations(),
            journal_before_update
        );
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        let consumer_saved = consumer.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("native consumer save")
                .clone()
        });
        let (consumer_container, _) = read_container(&consumer_saved);
        let consumer_engine = consumer_container
            .documents
            .first()
            .expect("actual consumer engine");
        assert_eq!(consumer_engine.dependencies, vec![second_reference.clone()]);
        assert_ne!(consumer_engine.document_id, second_reference.asset_id);
        assert_eq!(
            consumer_container.provenance["input_version"],
            serde_json::to_value(&second_reference).expect("exact version")
        );
        assert_eq!(
            consumer_container.provenance["producer_provenance"],
            second_version.provenance
        );
        assert_eq!(
            consumer_container.provenance["producer_editable_source"],
            serde_json::to_value(&second_version.editable_source).expect("exact producer source")
        );
        drop(session);
        close_artboard(&workspace, consumer, window).await;
        let modal = saved_sim_search(
            &workspace,
            &consumer_saved.identity.relative_path.to_string_lossy(),
            window,
        );
        modal.read_with(window, |modal, _| {
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("actual saved linked-output consumer reopen");
        let frame = reopened.read_with(window, |item, cx| {
            assert_eq!(item.input_version(cx), Some(&second_reference));
            assert_eq!(item.saved_sim_source(cx), Some(&consumer_saved));
            let session = item.session().read(cx);
            assert_eq!(session.revision(), revision + 1);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            assert_eq!(session.document().network(), &updated_network);
            assert_eq!(
                session
                    .image_placement()
                    .expect("restored placement")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            let preview = session.preview().expect("automatic reopened frame");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("exact reopened native pixels"),
                updated_frame
            );
            preview
                .image()
                .as_bytes(0)
                .expect("exact export native frame")
                .to_vec()
        });
        let mut rgba = frame;
        let (pixels, remainder) = rgba.as_chunks_mut::<4>();
        assert!(remainder.is_empty());
        for pixel in pixels {
            pixel.swap(0, 2);
        }
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        let final_saved = reopened.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Exported baked PNG:"))
            );
            item.saved_sim_source(cx)
                .expect("actual final consumer source")
                .clone()
        });
        let (final_container, _) = read_container(&final_saved);
        let consumer_output_reference = final_container
            .build_outputs
            .first()
            .expect("actual consumer derived output");
        assert_eq!(
            consumer_output_reference.asset_id,
            consumer_engine.document_id
        );
        let derived = fixture
            .assets
            .lock()
            .expect("actual new committed output")
            .resolve_project_asset_version(
                consumer_output_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("actual consumer output with historical producer dependencies/source closure");
        assert_eq!(
            derived.version().dependencies,
            vec![second_reference.clone()]
        );
        assert_eq!(
            derived.version().provenance["input_versions"],
            serde_json::json!([second_reference])
        );
        assert_eq!(
            derived.version().provenance["output"]["pixels_sha256"],
            format!("{:x}", Sha256::digest(&rgba))
        );
        assert_eq!(
            derived.version().provenance["output"]["encoded_sha256"],
            format!("{:x}", Sha256::digest(derived.bytes()))
        );
        let decoded = image::load_from_memory_with_format(derived.bytes(), image::ImageFormat::Png)
            .expect("actual committed PNG decode")
            .into_rgba8();
        assert_eq!(decoded.dimensions(), (1920_u32, 1080_u32));
        assert_eq!(decoded.as_raw(), &rgba);
        assert_eq!(
            derived
                .version()
                .stored_identity
                .relative_path
                .parent()
                .expect("actual declared destination"),
            Path::new("graphite").join(consumer_engine.document_id.to_string())
        );
        assert!(
            committer
                .lock()
                .expect("canonical commit journal")
                .operations()
                .iter()
                .any(|operation| {
                    operation.operation_id == consumer_output_reference.version_id
                        && operation.identity == derived.version().stored_identity
                        && operation.state == sim_runtime::OutputOperationState::Committed
                        && operation.sha256 == consumer_output_reference.content_sha256
                })
        );
        assert_eq!(read_container(&first_saved).1, first_saved_bytes);
        assert_eq!(read_container(&producer_source).1, producer_source_bytes);
        assert_eq!(
            read_container(&producer_latest_saved).1,
            producer_latest_bytes
        );
        let retained = fixture
            .assets
            .lock()
            .expect("original producer versions remain")
            .resolve_project_asset_version(
                &first_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("first immutable producer still usable");
        assert_eq!(retained.version(), &producer_version);
        assert_eq!(retained.bytes(), producer_bytes);
        let retained = fixture
            .assets
            .lock()
            .expect("successor producer version remains")
            .resolve_project_asset_version(
                &second_reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("second immutable producer still usable");
        assert_eq!(retained.version(), &second_version);
        assert_eq!(retained.bytes(), second_bytes);
        producer.read_with(window, |item, cx| {
            let session = item.session().read(cx);
            assert_eq!(session.revision(), producer_current_checkpoint.0);
            assert_eq!(session.parameters(), producer_current_checkpoint.1);
            assert_eq!(session.configuration(), producer_current_checkpoint.2);
            assert_eq!(session.document().network(), &producer_current_checkpoint.3);
            assert_ne!(session.document().network(), &producer_network);
            assert_eq!(item.input_version(cx), Some(&producer_input));
        });
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("unchanged original Input")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("authentic producer bytes"),
            png.encoded
        );
        close_artboard(&workspace, reopened, window).await;
        workspace.update_in(window, |workspace, window, cx| {
            assert!(workspace.activate_item(&producer, true, true, window, cx));
        });
        close_artboard(&workspace, producer, window).await;
    }

    #[gpui::test]
    async fn at_r28_published_png_native_missing_changed_source_stale_authority_cancel_and_retry_fail_closed(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real shared project service");
        let png = indexed_request_png(&fixture, 4).expect("actual original producer");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, _) = register_fixture_workflow(&fixture, &png, window);
        let producer = open_requested_png(&workspace, window);
        click(window, "GRAPHITE-ARTBOARD-EXPORT-PNG");
        let saved = producer.read_with(window, |item, cx| {
            item.saved_sim_source(cx)
                .expect("actual producer source")
                .clone()
        });
        let saved_bytes = fixture
            .assets
            .lock()
            .expect("real catalog")
            .read_verified(
                &saved.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("actual producer saved container");
        let container =
            sim_runtime::SimDocument::decode(&saved_bytes).expect("authentic container");
        let reference = container
            .build_outputs
            .first()
            .expect("actual derived reference")
            .clone();
        let published = fixture
            .assets
            .lock()
            .expect("real published version")
            .resolve_project_asset_version(
                &reference,
                &reader,
                &sim_tensor::CancellationToken::default(),
                64 * 1024 * 1024,
            )
            .expect("genuine committed producer dependency/source closure");
        let dependency = published
            .version()
            .dependencies
            .first()
            .expect("genuine producer dependency")
            .clone();
        let leaf = fixture
            .assets
            .lock()
            .expect("real retained dependency")
            .resolve_project_asset_version(
                &dependency,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("genuine producer Input version");
        let source = published
            .version()
            .editable_source
            .as_ref()
            .expect("actual producer editable source")
            .clone();
        let source_bytes = fixture
            .assets
            .lock()
            .expect("real source reader")
            .read_verified(
                &source.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("actual immutable producer source");
        let output_root = fixture
            .assets
            .lock()
            .expect("real service roots")
            .authorized_compatibility_folder_paths(&reader)
            .expect("genuine Output reader exposes the actual fixture root")
            .get(&sim_runtime::assets::AssetNamespace::Output)
            .expect("authorized test fixture Output root")
            .clone();
        let cases = [
            (
                published.version().stored_identity.clone(),
                published.bytes().to_vec(),
                true,
            ),
            (
                leaf.version().stored_identity.clone(),
                leaf.bytes().to_vec(),
                false,
            ),
            (source.identity.clone(), source_bytes.clone(), false),
        ];
        close_artboard(&workspace, producer, window).await;
        let modal = published_png_search(&workspace, &reference.version_id.to_string(), window);
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        modal.read_with(window, |modal, _| assert!(modal.diagnostic.is_none()));
        let consumer = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("actual linked consumer for saved-reopen authority controls");
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        let consumer_saved = consumer.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert_eq!(item.input_version(cx), Some(&reference));
            item.saved_sim_source(cx)
                .expect("real saved linked-output consumer")
                .clone()
        });
        close_artboard(&workspace, consumer, window).await;
        let journal_before = committer
            .lock()
            .expect("canonical journal")
            .operations()
            .to_vec();
        let versions_before = fixture
            .assets
            .lock()
            .expect("canonical catalog")
            .project_png_versions(&reader)
            .expect("registered version snapshot");
        for (identity, original, missing) in cases {
            let modal = published_png_search(&workspace, &reference.version_id.to_string(), window);
            let path = output_root.join(&identity.relative_path);
            if missing {
                std::fs::remove_file(&path).expect("actual missing selected Output version");
            } else {
                std::fs::write(&path, b"changed registered producer backing")
                    .expect("actual changed dependency/source bytes");
            }
            click(window, "GRAPHITE-PUBLISHED-PNG-ASSET-0");
            assert!(
                workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none())
            );
            modal.read_with(window, |modal, _| {
                assert!(!modal.opening);
                assert!(modal.open_cancellation.is_none());
                assert!(
                    modal
                        .diagnostic
                        .as_ref()
                        .is_some_and(|message| !message.is_empty())
                );
            });
            assert!(
                window
                    .debug_bounds("GRAPHITE-PUBLISHED-PNG-OPEN-ERROR")
                    .is_some()
            );
            assert_eq!(
                committer
                    .lock()
                    .expect("no publisher work on denied open")
                    .operations(),
                journal_before
            );
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("no pin or version on denied open")
                    .project_png_versions(&reader)
                    .expect("same registered declarations"),
                versions_before
            );
            std::fs::write(&path, &original)
                .expect("restore exact real fixture bytes for recovery");
            fixture
                .assets
                .lock()
                .expect("actual restored backing service")
                .scan_namespaces(
                    &[sim_runtime::assets::AssetNamespace::Output],
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                )
                .expect("canonical Output refresh after exact fixture restoration");
            click(window, "GRAPHITE-PUBLISHED-PNG-RETRY");
            modal.read_with(window, |modal, _| {
                assert!(modal.diagnostic.is_none());
                assert_eq!(modal.choices.len(), 1);
            });
            assert!(
                window
                    .debug_bounds("GRAPHITE-PUBLISHED-PNG-OPEN-ERROR")
                    .is_none()
            );
            click(window, "GRAPHITE-PUBLISHED-PNG-ASSET-0");
            modal.read_with(window, |modal, _| {
                assert!(
                    modal.diagnostic.is_none(),
                    "restored native open failed for {identity:?}: {:?}",
                    modal.diagnostic
                );
            });
            let recovered = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteArtboardItem>(cx)
                })
                .expect("actual native retry after restored backing");
            recovered.read_with(window, |item, cx| {
                assert_eq!(item.input_version(cx), Some(&reference));
                let session = item.session().read(cx);
                assert!(session.preview_is_current(session.preview().expect("real retry frame")));
            });
            close_artboard(&workspace, recovered, window).await;
        }
        let catalog_path = output_root.join(".sim/project-asset-versions.json");
        let original_catalog_bytes =
            std::fs::read(&catalog_path).expect("actual canonical fixture catalog");
        let original_catalog: serde_json::Value =
            serde_json::from_slice(&original_catalog_bytes).expect("actual stored catalog schema");
        for relabel_input in [true, false] {
            let mut tampered = original_catalog.clone();
            let version = tampered
                .get_mut("versions")
                .and_then(serde_json::Value::as_array_mut)
                .expect("actual versions array")
                .iter_mut()
                .find(|version| {
                    version["reference"]["version_id"] == serde_json::json!(reference.version_id)
                })
                .expect("actual selected published entry");
            if relabel_input {
                version["original_identity"] = serde_json::to_value(&png.asset_identity)
                    .expect("genuine Input identity used as negative relabel fixture");
            } else {
                version["reference"]["asset_id"] = serde_json::json!(uuid::Uuid::new_v4());
            }
            std::fs::write(
                &catalog_path,
                serde_json::to_vec(&tampered).expect("negative fixture catalog encoding"),
            )
            .expect("actual fixture catalog tamper");
            let modal = if relabel_input {
                let modal = saved_sim_search(
                    &workspace,
                    &consumer_saved.identity.relative_path.to_string_lossy(),
                    window,
                );
                modal.read_with(window, |modal, _| {
                    assert!(modal.diagnostic.is_none());
                    assert_eq!(modal.choices.len(), 1);
                });
                click(window, "GRAPHITE-SIM-ASSET-0");
                modal
            } else {
                let modal =
                    published_png_search(&workspace, &reference.version_id.to_string(), window);
                click(window, "GRAPHITE-PUBLISHED-PNG-ASSET-0");
                modal
            };
            modal.read_with(window, |modal, _| {
                assert!(!modal.opening);
                assert!(
                    modal
                        .diagnostic
                        .as_ref()
                        .is_some_and(|message| !message.is_empty())
                );
            });
            assert!(
                window
                    .debug_bounds(if relabel_input {
                        "GRAPHITE-SIM-OPEN-ERROR"
                    } else {
                        "GRAPHITE-PUBLISHED-PNG-OPEN-ERROR"
                    })
                    .is_some()
            );
            assert!(
                workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none())
            );
            assert_eq!(
                committer
                    .lock()
                    .expect("catalog tamper never creates publication")
                    .operations(),
                journal_before
            );
            std::fs::write(&catalog_path, &original_catalog_bytes)
                .expect("restore exact registered fixture catalog");
            assert_eq!(
                fixture
                    .assets
                    .lock()
                    .expect("restored catalog reader")
                    .project_png_versions(&reader)
                    .expect("exact restored registered versions"),
                versions_before
            );
            click(
                window,
                if relabel_input {
                    "GRAPHITE-SIM-RETRY"
                } else {
                    "GRAPHITE-PUBLISHED-PNG-RETRY"
                },
            );
            modal.read_with(window, |modal, _| {
                assert!(modal.diagnostic.is_none());
                assert_eq!(modal.choices.len(), 1);
            });
            click(
                window,
                if relabel_input {
                    "GRAPHITE-SIM-ASSET-0"
                } else {
                    "GRAPHITE-PUBLISHED-PNG-ASSET-0"
                },
            );
            let recovered = workspace
                .read_with(window, |workspace, cx| {
                    workspace.active_item_as::<GraphiteArtboardItem>(cx)
                })
                .expect("actual native recovery after exact catalog restore");
            recovered.read_with(window, |item, cx| {
                assert_eq!(item.input_version(cx), Some(&reference))
            });
            close_artboard(&workspace, recovered, window).await;
        }
        let modal = published_png_search(&workspace, &reference.version_id.to_string(), window);
        let weak_modal = modal.downgrade();
        click_without_pumping(window, "GRAPHITE-PUBLISHED-PNG-ASSET-0");
        let cancellation = modal.read_with(window, |modal, _| {
            assert!(modal.opening);
            assert!(modal.open_task.is_some());
            modal
                .open_cancellation
                .as_ref()
                .expect("actual caller-owned installation authority")
                .clone()
        });
        paint(window);
        click_without_pumping(window, "GRAPHITE-PUBLISHED-PNG-CANCEL");
        assert!(cancellation.is_cancelled());
        drop(modal);
        paint(window);
        window.run_until_parked();
        paint(window);
        assert!(weak_modal.upgrade().is_none());
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert!(workspace.read_with(window, |workspace, cx| {
            workspace.active_modal::<AuthorizedPngChooser>(cx).is_none()
        }));
        assert_eq!(
            committer
                .lock()
                .expect("cancelled linked open publishes nothing")
                .operations(),
            journal_before
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("cancelled linked open never pins")
                .project_png_versions(&reader)
                .expect("same immutable catalog"),
            versions_before
        );
        let modal = published_png_search(&workspace, &reference.version_id.to_string(), window);
        window.update(|_, cx| {
            crate::clear_asset_binding(cx).expect("genuine binding revocation before selected open")
        });
        click(window, "GRAPHITE-PUBLISHED-PNG-ASSET-0");
        modal.read_with(window, |modal, _| {
            assert!(!modal.opening);
            assert!(
                modal
                    .diagnostic
                    .as_ref()
                    .is_some_and(|message| !message.is_empty())
            );
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-PUBLISHED-PNG-OPEN-ERROR")
                .is_some()
        );
        assert!(workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none()));
        assert_eq!(
            committer
                .lock()
                .expect("stale authority publishes nothing")
                .operations(),
            journal_before
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("stale authority preserves versions")
                .project_png_versions(&reader)
                .expect("same immutable catalog"),
            versions_before
        );
    }

    #[gpui::test]
    async fn at_task573_saved_input_sim_reopens_after_genuine_pin_journal_eviction(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real canonical service");
        let png = indexed_request_png(&fixture, 4).expect("actual Input producer");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, committer, writer) = register_fixture_workflow(&fixture, &png, window);
        let roots = fixture
            .assets
            .lock()
            .expect("actual physical roots")
            .roots()
            .clone();
        {
            let mut publisher = committer.lock().expect("same shared canonical publisher");
            assert_eq!(publisher.operations().len(), 1);
            *publisher = sim_runtime::OutputCommitter::open_with_limits(
                roots.clone(),
                sim_runtime::DEFAULT_MAX_OUTPUT_BYTES,
                2,
                sim_runtime::DEFAULT_MAX_OUTPUT_JOURNAL_BYTES,
            )
            .expect("bounded journal admits the existing real operation");
        }
        let item = open_requested_png(&workspace, window);
        edit_artboard_field(&item, 9, "18", window);
        edit_artboard_field(&item, 10, "8", window);
        let (
            version,
            parameters,
            configuration,
            placement,
            revision,
            network,
            document_id,
            node_id,
            frame,
        ) = item.read_with(window, |item, cx| {
            let session = item.session().read(cx);
            let preview = session.preview().expect("actual native committed scene");
            assert!(session.preview_is_current(preview));
            (
                item.input_version(cx)
                    .expect("real retained Input version")
                    .clone(),
                session.parameters(),
                session.configuration(),
                session
                    .image_placement()
                    .expect("committed original placement"),
                session.revision(),
                session.document().network().clone(),
                session.document().document_id_encoding(),
                session.document().node_id(),
                preview.image().as_bytes(0).expect("actual frame").to_vec(),
            )
        });
        click(window, "GRAPHITE-ARTBOARD-SAVE-SIM");
        let saved = item.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            item.saved_sim_source(cx)
                .expect("actual editable .sim receipt")
                .clone()
        });
        let saved_bytes = fixture
            .assets
            .lock()
            .expect("real saved bytes")
            .read_verified(
                &saved.identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )
            .expect("canonical saved container");
        let retained_before = fixture
            .assets
            .lock()
            .expect("actual retained leaf")
            .resolve_project_asset_version(
                &version,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("genuine Input pin authority and bytes");
        assert_eq!(
            retained_before.version().original_identity,
            png.asset_identity
        );
        assert_eq!(retained_before.bytes(), png.encoded);
        close_artboard(&workspace, item, window).await;
        let versions_before = fixture
            .assets
            .lock()
            .expect("durable registered versions")
            .project_png_versions(&reader)
            .expect("exact catalog snapshot");
        let proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            "journal-eviction-regression",
            "png",
            0,
            png.descriptor.width,
            png.descriptor.height,
            png.encoded.clone(),
        )
        .expect("genuine additional PNG proposal");
        let receipt = {
            let mut assets = fixture.assets.lock().expect("same real asset service");
            let mut publisher = committer.lock().expect("same shared committer Arc");
            assert_eq!(publisher.operations().len(), 2);
            assert!(publisher.operation(version.version_id).is_some());
            *publisher = sim_runtime::OutputCommitter::open_with_limits(
                roots,
                sim_runtime::DEFAULT_MAX_OUTPUT_BYTES,
                2,
                sim_runtime::DEFAULT_MAX_OUTPUT_JOURNAL_BYTES,
            )
            .expect("bounded reopened journal admits both current operations");
            let receipts = publisher
                .commit_proposal_batch_and_register(
                    &[proposal],
                    "2026-10-10T12:00:00+00:00"
                        .parse()
                        .expect("fixed actual publication time"),
                    &mut assets,
                    &writer,
                    &sim_tensor::CancellationToken::default(),
                )
                .expect("one genuine publication evicts the oldest completed pin operation");
            assert_eq!(receipts.len(), 1);
            assert_eq!(publisher.operations().len(), 2);
            assert!(publisher.operation(version.version_id).is_none());
            receipts
                .into_iter()
                .next()
                .expect("authentic new committed fixture receipt")
        };
        assert_eq!(
            receipt.operation().state,
            sim_runtime::OutputOperationState::Committed
        );
        let journal_after_eviction = committer
            .lock()
            .expect("actual post-eviction journal")
            .operations()
            .to_vec();
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("durable catalog survives journal rotation")
                .project_png_versions(&reader)
                .expect("unchanged version declarations"),
            versions_before
        );
        let modal = saved_sim_search(
            &workspace,
            &saved.identity.relative_path.to_string_lossy(),
            window,
        );
        modal.read_with(window, |modal, _| {
            assert!(modal.diagnostic.is_none());
            assert_eq!(modal.choices.len(), 1);
        });
        window.simulate_keystrokes("enter");
        window.run_until_parked();
        paint(window);
        assert!(window.debug_bounds("GRAPHITE-SIM-OPEN-ERROR").is_none());
        let reopened = workspace
            .read_with(window, |workspace, cx| {
                workspace.active_item_as::<GraphiteArtboardItem>(cx)
            })
            .expect("actual saved Input .sim reopens without an evicted pin journal operation");
        reopened.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert_eq!(item.input_version(cx), Some(&version));
            assert_eq!(item.saved_sim_source(cx), Some(&saved));
            let session = item.session().read(cx);
            assert_eq!(session.revision(), revision);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            assert_eq!(
                session
                    .image_placement()
                    .expect("exact editable placement")
                    .to_cols_array()
                    .map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(session.document().document_id_encoding(), document_id);
            assert_eq!(session.document().node_id(), node_id);
            assert_eq!(session.document().network(), &network);
            let source = session
                .document()
                .encoded_png_input()
                .expect("original retained source")
                .source();
            assert_eq!(
                source.identity().version_id.as_ref(),
                version.version_id.to_string()
            );
            let preview = session.preview().expect("actual restored visible frame");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                preview
                    .image()
                    .as_bytes(0)
                    .expect("unchanged native result"),
                frame
            );
        });
        let retained_after = fixture
            .assets
            .lock()
            .expect("still registered immutable leaf")
            .resolve_project_asset_version(
                &version,
                &reader,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect("durable Input lease does not require live publication journal entry");
        assert_eq!(retained_after.version(), retained_before.version());
        assert_eq!(retained_after.bytes(), retained_before.bytes());
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("saved source stays immutable")
                .read_verified(
                    &saved.identity,
                    &reader,
                    &sim_tensor::CancellationToken::default(),
                    sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
                )
                .expect("original saved .sim remains intact"),
            saved_bytes
        );
        assert!(
            committer
                .lock()
                .expect("reopen causes no pin publication")
                .operation(version.version_id)
                .is_none()
        );
        assert_eq!(
            committer
                .lock()
                .expect("bounded journal remains unchanged")
                .operations()
                .len(),
            2
        );
        assert_eq!(
            committer
                .lock()
                .expect("reopen creates no journal operations")
                .operations(),
            journal_after_eviction
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("reopen retains exact catalog declarations")
                .project_png_versions(&reader)
                .expect("unchanged registered Input version"),
            versions_before
        );
        close_artboard(&workspace, reopened, window).await;
    }
}
