use std::{error::Error, fmt, rc::Rc};

use gpui::{AnyView, App, Context, Entity, EntityId, FocusHandle, Render, Role};
use project::Project;
use ui::{Color, Label, LabelSize, Window, prelude::*};

use crate::collaborative_accessibility::COMPOSER_LABEL;

type ComposerAction = Rc<dyn Fn(&mut App) -> Result<(), CollaborativeComposerActionError>>;

#[derive(Clone)]
pub struct CollaborativeComposerProvider {
    project: Entity<Project>,
    view: AnyView,
    footer: Option<AnyView>,
    input_in_timeline: bool,
    focus_handle: Option<FocusHandle>,
    submit: ComposerAction,
    cancel: ComposerAction,
}

impl CollaborativeComposerProvider {
    pub fn new(
        project: Entity<Project>,
        view: AnyView,
        submit: impl Fn(&mut App) -> Result<(), CollaborativeComposerActionError> + 'static,
        cancel: impl Fn(&mut App) -> Result<(), CollaborativeComposerActionError> + 'static,
    ) -> Self {
        Self {
            project,
            view,
            footer: None,
            input_in_timeline: false,
            focus_handle: None,
            submit: Rc::new(submit),
            cancel: Rc::new(cancel),
        }
    }

    pub fn with_focus_handle(mut self, focus_handle: FocusHandle) -> Self {
        self.focus_handle = Some(focus_handle);
        self
    }

    pub fn with_input_in_timeline(mut self) -> Self {
        self.input_in_timeline = true;
        self
    }

    pub fn input_in_timeline(&self) -> bool {
        self.input_in_timeline
    }

    pub fn with_footer(mut self, footer: AnyView) -> Self {
        self.footer = Some(footer);
        self
    }

    pub fn footer(&self) -> Option<AnyView> {
        self.footer.clone()
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn view(&self) -> &AnyView {
        &self.view
    }

    pub fn focus_handle(&self) -> Option<FocusHandle> {
        self.focus_handle.clone()
    }

    pub fn submit(&self, cx: &mut App) -> Result<(), CollaborativeComposerActionError> {
        (self.submit)(cx)
    }

    pub fn cancel(&self, cx: &mut App) -> Result<(), CollaborativeComposerActionError> {
        (self.cancel)(cx)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeComposerRegistration {
    provider_id: EntityId,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeComposerRegistrationError {
    ProjectMismatch,
    ProviderOccupied,
    RegistrationExhausted,
}

impl fmt::Display for CollaborativeComposerRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProjectMismatch => {
                formatter.write_str("composer provider belongs to a different project")
            }
            Self::ProviderOccupied => formatter.write_str("composer provider is already occupied"),
            Self::RegistrationExhausted => {
                formatter.write_str("composer registration generation is exhausted")
            }
        }
    }
}

impl Error for CollaborativeComposerRegistrationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeComposerActionError {
    ThreadUnavailable,
    EmptyInput,
    ProviderFailure(String),
}

impl fmt::Display for CollaborativeComposerActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadUnavailable => formatter.write_str("no active agent thread is available"),
            Self::EmptyInput => formatter.write_str("message input is empty"),
            Self::ProviderFailure(message) => {
                write!(formatter, "composer action failed: {message}")
            }
        }
    }
}

impl Error for CollaborativeComposerActionError {}

pub struct CollaborativeComposerHost {
    project: Entity<Project>,
    provider: Option<CollaborativeComposerProvider>,
    provider_generation: Option<u64>,
    next_generation: u64,
}

impl CollaborativeComposerHost {
    pub fn new(project: Entity<Project>) -> Self {
        Self {
            project,
            provider: None,
            provider_generation: None,
            next_generation: 0,
        }
    }

    pub fn register(
        &mut self,
        provider: CollaborativeComposerProvider,
    ) -> Result<CollaborativeComposerRegistration, CollaborativeComposerRegistrationError> {
        if provider.project.entity_id() != self.project.entity_id() {
            return Err(CollaborativeComposerRegistrationError::ProjectMismatch);
        }
        if self.provider.is_some() {
            return Err(CollaborativeComposerRegistrationError::ProviderOccupied);
        }
        let generation = self
            .next_generation
            .checked_add(1)
            .ok_or(CollaborativeComposerRegistrationError::RegistrationExhausted)?;
        self.next_generation = generation;
        let registration = CollaborativeComposerRegistration {
            provider_id: provider.view.entity_id(),
            generation,
        };
        self.provider = Some(provider);
        self.provider_generation = Some(generation);
        Ok(registration)
    }

    pub fn unregister(&mut self, registration: CollaborativeComposerRegistration) -> bool {
        if !self
            .provider
            .as_ref()
            .is_some_and(|provider| provider.view.entity_id() == registration.provider_id)
            || self.provider_generation != Some(registration.generation)
        {
            return false;
        }
        self.provider = None;
        self.provider_generation = None;
        true
    }

    pub fn input_in_timeline(&self) -> bool {
        self.provider
            .as_ref()
            .is_some_and(CollaborativeComposerProvider::input_in_timeline)
    }

    pub fn view(&self) -> Option<AnyView> {
        self.provider.as_ref().map(|provider| provider.view.clone())
    }

    pub fn footer(&self) -> Option<AnyView> {
        self.provider
            .as_ref()
            .and_then(CollaborativeComposerProvider::footer)
    }

    pub fn focus_handle(&self) -> Option<FocusHandle> {
        self.provider
            .as_ref()
            .and_then(CollaborativeComposerProvider::focus_handle)
    }

    pub fn submit(&self, cx: &mut App) -> Result<(), CollaborativeComposerActionError> {
        let provider = self
            .provider
            .as_ref()
            .ok_or(CollaborativeComposerActionError::ThreadUnavailable)?;
        provider.submit(cx)
    }

    pub fn cancel(&self, cx: &mut App) -> Result<(), CollaborativeComposerActionError> {
        let provider = self
            .provider
            .as_ref()
            .ok_or(CollaborativeComposerActionError::ThreadUnavailable)?;
        provider.cancel(cx)
    }
}

pub(crate) struct CollaborativeComposerSurface {
    view: Option<AnyView>,
    unavailable_message: Option<gpui::SharedString>,
    focus_handle: FocusHandle,
    provider_focus_handle: Option<FocusHandle>,
    input_in_timeline: bool,
}

impl CollaborativeComposerSurface {
    pub(crate) fn new(cx: &mut Context<Self>) -> Self {
        Self {
            view: None,
            unavailable_message: None,
            focus_handle: cx.focus_handle().tab_stop(true),
            provider_focus_handle: None,
            input_in_timeline: false,
        }
    }

    pub(crate) fn set_view(
        &mut self,
        view: Option<AnyView>,
        provider_focus_handle: Option<FocusHandle>,
        unavailable_message: Option<gpui::SharedString>,
        input_in_timeline: bool,
        cx: &mut Context<Self>,
    ) {
        let changed = self.view.as_ref().map(AnyView::entity_id)
            != view.as_ref().map(AnyView::entity_id)
            || self.provider_focus_handle != provider_focus_handle
            || self.unavailable_message != unavailable_message
            || self.input_in_timeline != input_in_timeline;
        if changed {
            self.view = view;
            self.provider_focus_handle = provider_focus_handle;
            self.unavailable_message = unavailable_message;
            self.input_in_timeline = input_in_timeline;
            cx.notify();
        }
    }

    pub(crate) fn input_in_timeline(&self) -> bool {
        self.view.is_some() && self.input_in_timeline
    }

    pub(crate) fn focus_handle(&self) -> FocusHandle {
        self.provider_focus_handle
            .clone()
            .unwrap_or_else(|| self.focus_handle.clone())
    }
}

impl Render for CollaborativeComposerSurface {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view_available = self.view.is_some();
        v_flex()
            .id("collaborative-composer")
            .debug_selector(|| "COLLABORATIVE-COMPOSER".to_owned())
            .w_full()
            .flex_none()
            .track_focus(&self.focus_handle)
            .tab_index(0)
            .role(Role::Group)
            .aria_label(COMPOSER_LABEL)
            .border_t_1()
            .border_color(cx.theme().colors().border)
            .bg(cx.theme().colors().background)
            .when_some(self.view.clone(), |this, view| {
                this.child(
                    div()
                        .id("collaborative-composer-editor")
                        .debug_selector(|| "COLLABORATIVE-COMPOSER-EDITOR".to_owned())
                        .w_full()
                        .child(view),
                )
            })
            .when(!view_available, |this| {
                this.py(DynamicSpacing::Base16.rems(cx))
                    .items_center()
                    .justify_center()
                    .child(
                        Label::new(
                            self.unavailable_message
                                .clone()
                                .unwrap_or_else(|| "Select a task to start collaborating".into()),
                        )
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    )
            })
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, path::Path, rc::Rc};

    use fs::FakeFs;
    use gpui::{Empty, TestAppContext};
    use settings::SettingsStore;

    use super::*;

    #[gpui::test]
    async fn collaborative_composer(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.set_global(db::AppDatabase::test_new());
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let file_system = FakeFs::new(cx.executor());
        let project = Project::test(file_system.clone(), [Path::new("/project")], cx).await;
        let other_project = Project::test(file_system, [Path::new("/other")], cx).await;
        let provider_view = cx.new(|_| Empty);
        let submitted = Rc::new(Cell::new(0));
        let cancelled = Rc::new(Cell::new(0));
        let mut host = CollaborativeComposerHost::new(project.clone());

        assert_eq!(
            cx.update(|cx| host.submit(cx)),
            Err(CollaborativeComposerActionError::ThreadUnavailable)
        );
        assert_eq!(
            cx.update(|cx| host.cancel(cx)),
            Err(CollaborativeComposerActionError::ThreadUnavailable)
        );

        let mismatch = CollaborativeComposerProvider::new(
            other_project,
            provider_view.clone().into(),
            |_| Ok(()),
            |_| Ok(()),
        );
        assert_eq!(
            host.register(mismatch),
            Err(CollaborativeComposerRegistrationError::ProjectMismatch)
        );

        let submitted_for_provider = submitted.clone();
        let cancelled_for_provider = cancelled.clone();
        let provider = CollaborativeComposerProvider::new(
            project,
            provider_view.clone().into(),
            move |_| {
                submitted_for_provider.set(submitted_for_provider.get() + 1);
                Ok(())
            },
            move |_| {
                cancelled_for_provider.set(cancelled_for_provider.get() + 1);
                Ok(())
            },
        );
        let registration = host
            .register(provider)
            .expect("canonical composer provider should register");
        assert_eq!(
            host.view()
                .expect("registered composer should expose its native view")
                .entity_id(),
            provider_view.entity_id()
        );
        cx.update(|cx| host.submit(cx))
            .expect("submit should route to the provider");
        cx.update(|cx| host.cancel(cx))
            .expect("cancel should route to the provider");
        assert_eq!((submitted.get(), cancelled.get()), (1, 1));

        let occupied = CollaborativeComposerProvider::new(
            host.project.clone(),
            cx.new(|_| Empty).into(),
            |_| Err(CollaborativeComposerActionError::EmptyInput),
            |_| Ok(()),
        );
        assert_eq!(
            host.register(occupied),
            Err(CollaborativeComposerRegistrationError::ProviderOccupied)
        );
        assert!(host.unregister(registration));

        let empty_provider = CollaborativeComposerProvider::new(
            host.project.clone(),
            provider_view.into(),
            |_| Err(CollaborativeComposerActionError::EmptyInput),
            |_| Ok(()),
        );
        host.register(empty_provider)
            .expect("replacement composer should register");
        assert_eq!(
            cx.update(|cx| host.submit(cx)),
            Err(CollaborativeComposerActionError::EmptyInput)
        );
        assert!(!host.unregister(registration));
    }
}
