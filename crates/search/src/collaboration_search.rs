use gpui::{
    App, Context, EventEmitter, FocusHandle, Focusable, IntoElement, Render, SharedString, Window,
};
use menu::Confirm;
use ui::{Color, Label, LabelSize, ListItem, prelude::*};
use zed_actions::search::{SelectNextMatch, SelectPreviousMatch};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CollaborationResultGroup {
    Community,
    Channel,
    Member,
    Project,
    Message,
    Repository,
    Task,
    Agent,
    Workflow,
    Media,
}

impl CollaborationResultGroup {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Community => "Communities",
            Self::Channel => "Channels",
            Self::Member => "Members",
            Self::Project => "Collaborative Projects",
            Self::Message => "Messages",
            Self::Repository => "Repositories",
            Self::Task => "Tasks",
            Self::Agent => "Agents",
            Self::Workflow => "Workflows",
            Self::Media => "Media",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NativeResultGroup {
    File,
    Project,
}

impl NativeResultGroup {
    pub const fn label(self) -> &'static str {
        match self {
            Self::File => "Files",
            Self::Project => "Projects",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SearchResultGroup {
    Native(NativeResultGroup),
    Collaboration(CollaborationResultGroup),
}

impl SearchResultGroup {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Native(group) => group.label(),
            Self::Collaboration(group) => group.label(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum SearchResultIdentity {
    Native(SharedString),
    Collaboration(SharedString),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchPresentationItem {
    pub identity: SearchResultIdentity,
    pub group: SearchResultGroup,
    pub label: SharedString,
    pub detail: Option<SharedString>,
}

impl SearchPresentationItem {
    pub fn native(
        identity: impl Into<SharedString>,
        group: NativeResultGroup,
        label: impl Into<SharedString>,
        detail: Option<SharedString>,
    ) -> Self {
        Self {
            identity: SearchResultIdentity::Native(identity.into()),
            group: SearchResultGroup::Native(group),
            label: label.into(),
            detail,
        }
    }

    pub fn collaboration(
        identity: impl Into<SharedString>,
        group: CollaborationResultGroup,
        label: impl Into<SharedString>,
        detail: Option<SharedString>,
    ) -> Self {
        Self {
            identity: SearchResultIdentity::Collaboration(identity.into()),
            group: SearchResultGroup::Collaboration(group),
            label: label.into(),
            detail,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationSearchFreshness {
    Current,
    Lagging { affected_checkpoints: u64 },
    Unavailable,
}

impl CollaborationSearchFreshness {
    pub fn label(self) -> SharedString {
        match self {
            Self::Current => "Current".into(),
            Self::Lagging {
                affected_checkpoints: 1,
            } => "Results may be stale · 1 source behind".into(),
            Self::Lagging {
                affected_checkpoints,
            } => format!("Results may be stale · {affected_checkpoints} sources behind").into(),
            Self::Unavailable => "Freshness unavailable".into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationSearchPresentation {
    Authorized {
        scope_label: SharedString,
        freshness: CollaborationSearchFreshness,
        items: Vec<SearchPresentationItem>,
    },
    Unauthorized,
}

impl CollaborationSearchPresentation {
    pub fn authorized(
        scope_label: impl Into<SharedString>,
        freshness: CollaborationSearchFreshness,
        items: Vec<SearchPresentationItem>,
    ) -> Self {
        Self::Authorized {
            scope_label: scope_label.into(),
            freshness,
            items,
        }
    }

    pub fn unauthorized() -> Self {
        Self::Unauthorized
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationSearchEvent {
    Confirmed(SearchResultIdentity),
}

pub struct CollaborationSearchView {
    focus_handle: FocusHandle,
    native_items: Vec<SearchPresentationItem>,
    collaboration: CollaborationSearchPresentation,
    ordered_items: Vec<SearchPresentationItem>,
    selected_index: Option<usize>,
}

impl CollaborationSearchView {
    pub fn new(
        native_items: Vec<SearchPresentationItem>,
        collaboration: CollaborationSearchPresentation,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut this = Self {
            focus_handle: cx.focus_handle(),
            native_items,
            collaboration,
            ordered_items: Vec::new(),
            selected_index: None,
        };
        this.rebuild_items(None);
        this
    }

    pub fn update_results(
        &mut self,
        native_items: Vec<SearchPresentationItem>,
        collaboration: CollaborationSearchPresentation,
        cx: &mut Context<Self>,
    ) {
        let selected_identity = self.selected_item().map(|item| item.identity.clone());
        self.native_items = native_items;
        self.collaboration = collaboration;
        self.rebuild_items(selected_identity.as_ref());
        cx.notify();
    }

    pub fn selected_item(&self) -> Option<&SearchPresentationItem> {
        self.selected_index
            .and_then(|index| self.ordered_items.get(index))
    }

    pub fn ordered_items(&self) -> &[SearchPresentationItem] {
        &self.ordered_items
    }

    pub fn collaboration_status_label(&self) -> SharedString {
        match &self.collaboration {
            CollaborationSearchPresentation::Authorized {
                scope_label, items, ..
            } if items.is_empty() => format!("No collaboration results in {scope_label}").into(),
            CollaborationSearchPresentation::Authorized {
                scope_label,
                freshness,
                ..
            } => format!("{scope_label} · {}", freshness.label()).into(),
            CollaborationSearchPresentation::Unauthorized => {
                "Collaboration results unavailable".into()
            }
        }
    }

    fn rebuild_items(&mut self, selected_identity: Option<&SearchResultIdentity>) {
        let mut items = self
            .native_items
            .iter()
            .filter(|item| matches!(item.group, SearchResultGroup::Native(_)))
            .cloned()
            .collect::<Vec<_>>();
        if let CollaborationSearchPresentation::Authorized {
            items: collaboration_items,
            ..
        } = &self.collaboration
        {
            items.extend(
                collaboration_items
                    .iter()
                    .filter(|item| matches!(item.group, SearchResultGroup::Collaboration(_)))
                    .cloned(),
            );
        }
        items.sort_by_key(|item| item.group);
        self.ordered_items = items;
        self.selected_index = selected_identity
            .and_then(|identity| {
                self.ordered_items
                    .iter()
                    .position(|item| &item.identity == identity)
            })
            .or_else(|| (!self.ordered_items.is_empty()).then_some(0));
    }

    fn select_next(&mut self, _: &SelectNextMatch, _: &mut Window, cx: &mut Context<Self>) {
        if self.ordered_items.is_empty() {
            return;
        }
        self.selected_index = Some(match self.selected_index {
            Some(index) => (index + 1) % self.ordered_items.len(),
            None => 0,
        });
        cx.stop_propagation();
        cx.notify();
    }

    fn select_previous(&mut self, _: &SelectPreviousMatch, _: &mut Window, cx: &mut Context<Self>) {
        if self.ordered_items.is_empty() {
            return;
        }
        self.selected_index = Some(match self.selected_index {
            Some(0) | None => self.ordered_items.len() - 1,
            Some(index) => index - 1,
        });
        cx.stop_propagation();
        cx.notify();
    }

    fn confirm(&mut self, _: &Confirm, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(item) = self.selected_item() {
            cx.stop_propagation();
            cx.emit(CollaborationSearchEvent::Confirmed(item.identity.clone()));
        }
    }

    fn render_items(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut rendered = Vec::new();
        let mut previous_group = None;
        for (index, item) in self.ordered_items.iter().enumerate() {
            if previous_group != Some(item.group) {
                previous_group = Some(item.group);
                rendered.push(
                    Label::new(item.group.label())
                        .size(LabelSize::Small)
                        .color(Color::Muted)
                        .into_any_element(),
                );
            }
            let detail = item.detail.clone();
            rendered.push(
                ListItem::new(("collaboration-search-result", index))
                    .toggle_state(self.selected_index == Some(index))
                    .inset(true)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.selected_index = Some(index);
                        cx.notify();
                    }))
                    .child(
                        h_flex()
                            .w_full()
                            .min_w_0()
                            .gap_2()
                            .child(Label::new(item.label.clone()))
                            .when_some(detail, |this, detail| {
                                this.child(
                                    Label::new(detail)
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                )
                            }),
                    )
                    .into_any_element(),
            );
        }
        rendered
    }
}

impl EventEmitter<CollaborationSearchEvent> for CollaborationSearchView {}

impl Focusable for CollaborationSearchView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for CollaborationSearchView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("collaboration-search-results")
            .key_context("CollaborationSearch")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::select_next))
            .on_action(cx.listener(Self::select_previous))
            .on_action(cx.listener(Self::confirm))
            .size_full()
            .gap_1()
            .child(
                Label::new(self.collaboration_status_label())
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
            .children(self.render_items(cx))
    }
}
