use anyhow::Result;
use gpui::{App, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, IconOrSvg, InlineDescription, LanguageModel, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    ModelDiscoveryStatus, ProviderSettingsView,
};
pub use openai_subscribed::PROVIDER_ID;
use openai_subscribed::{PROVIDER_NAME, State};
use std::sync::Arc;
use ui::{ButtonLink, prelude::*};

const SUBSCRIPTION_DESCRIPTION: &str = "Use your ChatGPT plan for eligible requests.";

pub struct OpenAiSubscribedProvider {
    state: Entity<State>,
}

impl OpenAiSubscribedProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        Self {
            state: cx.new(|cx| State::new(http_client, cx)),
        }
    }
}

impl LanguageModelProviderState for OpenAiSubscribedProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiSubscribedProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiGptSub)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.provided_models(cx).into_iter().next()
    }

    fn default_fast_model(&self, _: &App) -> Option<Arc<dyn LanguageModel>> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        State::account_models(&self.state, cx)
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).has_chatgpt_plan_permission()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        Task::ready(
            self.is_authenticated(cx)
                .then_some(())
                .ok_or(AuthenticateError::CredentialsNotFound),
        )
    }

    fn model_discovery_status(&self, cx: &App) -> Option<ModelDiscoveryStatus> {
        let state = self.state.read(cx);
        Some(
            if state.chatgpt_plan_busy() || state.chatgpt_catalog_loading() {
                ModelDiscoveryStatus::Loading
            } else if let Some(error) = state.chatgpt_catalog_error() {
                ModelDiscoveryStatus::Failed {
                    message: error,
                    can_retry: state.has_chatgpt_plan_permission(),
                }
            } else if !state.has_chatgpt_plan_permission() {
                ModelDiscoveryStatus::SignIn
            } else if !state.has_chatgpt_catalog() {
                ModelDiscoveryStatus::Empty
            } else {
                return None;
            },
        )
    }

    fn refresh_models(&self, cx: &mut App) {
        self.state
            .update(cx, |state, cx| state.refresh_chatgpt_catalog(cx));
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.is_authenticated(cx);
        let title = if is_authenticated {
            None
        } else {
            Some("Configure ChatGPT".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(SUBSCRIPTION_DESCRIPTION.into()))
        };

        Some(ProviderSettingsView::Inline(
            language_model::InlineProviderSettings {
                title,
                description,
                create_view: Arc::new({
                    let state = self.state.clone();
                    move |_window, cx| {
                        cx.new(|cx| ConfigurationView::new(state.clone(), cx))
                            .into()
                    }
                }),
            },
        ))
    }

    fn authentication_error_message(&self) -> SharedString {
        "Your ChatGPT subscription session is invalid or has expired. \
        Sign in again via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn missing_credentials_error_message(&self) -> SharedString {
        "You are not signed in to your ChatGPT account. \
        Sign in via Settings > AI > LLM Providers to continue."
            .into()
    }
}

struct ConfigurationView {
    state: Entity<State>,
    _subscription: gpui::Subscription,
}

impl ConfigurationView {
    fn new(state: Entity<State>, cx: &mut Context<Self>) -> Self {
        let subscription = cx.observe(&state, |_, _, cx| cx.notify());
        Self {
            state,
            _subscription: subscription,
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let profiles = state.chatgpt_profiles();
        let active = profiles.iter().find(|account| account.selected);
        let busy = state.chatgpt_plan_busy();
        let refreshing = state.chatgpt_catalog_loading();

        v_flex()
            .gap_2()
            .child(
                Button::new(
                    "chatgpt-plan-sign-in",
                    if profiles.is_empty() {
                        "Continue with ChatGPT"
                    } else {
                        "Add ChatGPT Account"
                    },
                )
                .start_icon(Icon::new(IconName::AiOpenAiGptSub))
                .style(ButtonStyle::Outlined)
                .disabled(busy)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.state
                        .update(cx, |state, cx| state.sign_in_chatgpt_plan(None, cx));
                })),
            )
            .children(profiles.iter().map(|account| {
                let id = account.id.clone();
                let label = match (account.selected, account.can_use_plan) {
                    (true, true) => format!("{} · Using ChatGPT plan", account.label),
                    (true, false) if account.connected => {
                        format!("{} · Plan usage disabled", account.label)
                    }
                    _ => account.label.clone(),
                };
                let action = if !account.connected {
                    "Sign In"
                } else if account.selected && !account.can_use_plan {
                    "Enable Plan Usage"
                } else if account.selected && state.chatgpt_catalog_error().is_some() {
                    "Reconnect"
                } else if account.selected {
                    "Selected"
                } else {
                    "Switch"
                };
                let reconnect = action == "Reconnect";
                h_flex().gap_2().child(Label::new(label)).child(
                    Button::new(format!("chatgpt-account-{}", account.id), action)
                        .disabled(busy || action == "Selected")
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.state.update(cx, |state, cx| {
                                if reconnect {
                                    state.sign_in_chatgpt_plan(Some(id.clone()), cx);
                                } else {
                                    state.select_chatgpt_account(id.clone(), cx);
                                }
                            });
                        })),
                )
            }))
            .when_some(
                active.filter(|account| account.connected),
                |view, account| {
                    view.child(
                        h_flex()
                            .gap_2()
                            .when(account.can_use_plan, |row| {
                                row.child(
                                    Button::new(
                                        "chatgpt-refresh-models",
                                        if refreshing {
                                            "Refreshing Models…"
                                        } else {
                                            "Refresh Models"
                                        },
                                    )
                                    .disabled(busy || refreshing)
                                    .on_click(cx.listener(
                                        |this, _, _, cx| {
                                            this.state.update(cx, |state, cx| {
                                                state.refresh_chatgpt_catalog(cx)
                                            });
                                        },
                                    )),
                                )
                                .child(ButtonLink::new(
                                    "Manage Usage",
                                    "https://chatgpt.com/#settings/usage",
                                ))
                            })
                            .child(
                                Button::new("chatgpt-plan-sign-out", "Sign Out")
                                    .disabled(busy)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.state.update(cx, |state, cx| {
                                            state.sign_out_chatgpt_plan(cx)
                                        });
                                    })),
                            ),
                    )
                },
            )
            .when_some(state.chatgpt_catalog_error(), |view, error| {
                view.child(Label::new(error).color(Color::Error))
            })
            .when_some(state.chatgpt_plan_error(), |view, error| {
                view.child(Label::new(error).color(Color::Error))
            })
            .when(
                state.has_chatgpt_plan_permission()
                    && !refreshing
                    && !state.has_chatgpt_catalog()
                    && state.chatgpt_catalog_error().is_none(),
                |view| {
                    view.child(
                        Label::new("No models are available for this ChatGPT account.")
                            .color(Color::Muted),
                    )
                },
            )
    }
}
