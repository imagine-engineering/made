use std::sync::Arc;

use client::{Client, UserStore};
use gpui::{AnyView, Entity, IntoElement, ParentElement};
use language_model::{
    LanguageModelProviderId, LanguageModelRegistry, ProviderSettingsView, ZED_CLOUD_PROVIDER_ID,
};
use ui::prelude::*;

use crate::{AgentPanelOnboardingCard, ApiKeysWithoutProviders, ZedAiOnboarding};

pub struct AgentPanelOnboarding {
    local_setup: Option<AnyView>,
    show_cloud_setup: bool,
    setup_error: Option<String>,
    user_store: Entity<UserStore>,
    client: Arc<Client>,
    has_configured_providers: bool,
    continue_with_zed_ai: Arc<dyn Fn(&mut Window, &mut App)>,
}

impl AgentPanelOnboarding {
    pub fn new(
        user_store: Entity<UserStore>,
        client: Arc<Client>,
        continue_with_zed_ai: impl Fn(&mut Window, &mut App) + 'static,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.subscribe(
            &LanguageModelRegistry::global(cx),
            |this: &mut Self, _registry, event: &language_model::Event, cx| match event {
                language_model::Event::ProviderStateChanged(_)
                | language_model::Event::AddedProvider(_)
                | language_model::Event::RemovedProvider(_)
                | language_model::Event::ProvidersChanged => {
                    this.has_configured_providers = Self::has_configured_providers(cx);
                    cx.notify();
                }
                _ => {}
            },
        )
        .detach();

        Self {
            local_setup: None,
            show_cloud_setup: false,
            setup_error: None,
            user_store,
            client,
            has_configured_providers: Self::has_configured_providers(cx),
            continue_with_zed_ai: Arc::new(continue_with_zed_ai),
        }
    }

    fn has_configured_providers(cx: &App) -> bool {
        LanguageModelRegistry::read_global(cx)
            .visible_providers()
            .iter()
            .any(|provider| provider.is_authenticated(cx) && provider.id() != ZED_CLOUD_PROVIDER_ID)
    }
}

impl Render for AgentPanelOnboarding {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let onboarding = ZedAiOnboarding::new(
            self.client.clone(),
            &self.user_store,
            self.continue_with_zed_ai.clone(),
            cx,
        )
        .with_dismiss({
            let callback = self.continue_with_zed_ai.clone();
            move |window, cx| callback(window, cx)
        });

        AgentPanelOnboardingCard::new()
            .child(Headline::new("Choose how to run your native agent").size(HeadlineSize::Small))
            .child(
                h_flex().gap_2()
                    .child(Button::new("choose-local-models", "Local models")
                        .style(ButtonStyle::Outlined).toggle_state(self.local_setup.is_some())
                        .selected_style(ButtonStyle::Tinted(ui::TintColor::Accent))
                        .on_click(cx.listener(|this, _, window, cx| {
                            let provider = LanguageModelRegistry::read_global(cx)
                                .provider(&LanguageModelProviderId::new("llama.cpp"));
                            match provider.and_then(|provider| provider.settings_view(cx)) {
                                Some(ProviderSettingsView::SubPage(settings)) => {
                                    this.local_setup = Some((settings.create_view)(window, cx));
                                    this.show_cloud_setup = false;
                                    this.setup_error = None;
                                }
                                _ => this.setup_error = Some("Local model setup is unavailable. Open model-provider settings and retry.".into()),
                            }
                            cx.notify();
                        })))
                    .child(Button::new("choose-cloud-providers", "Cloud providers")
                        .style(ButtonStyle::Outlined).toggle_state(self.show_cloud_setup)
                        .selected_style(ButtonStyle::Tinted(ui::TintColor::Accent))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.local_setup = None;
                            this.show_cloud_setup = true;
                            this.setup_error = None;
                            cx.notify();
                        }))),
            )
            .when(self.has_configured_providers, |this| {
                this.child(Button::new("continue-native-agent", "Continue with selected model").style(ButtonStyle::Filled).on_click({
                    let callback = self.continue_with_zed_ai.clone();
                    move |_, window, cx| callback(window, cx)
                }))
            })
            .children(self.local_setup.clone())
            .when(self.show_cloud_setup, |this| this.child(onboarding).child(ApiKeysWithoutProviders::new()))
            .when_some(self.setup_error.clone(), |this, message| this.child(Label::new(message).color(Color::Error)))
    }
}
