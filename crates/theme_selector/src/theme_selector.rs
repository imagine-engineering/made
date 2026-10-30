mod icon_theme_selector;

use fs::Fs;
use fuzzy::{StringMatch, StringMatchCandidate, match_strings};
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, Focusable, Render, UpdateGlobal, WeakEntity,
    Window, actions,
};
use picker::{Picker, PickerDelegate};
use settings::{ActiveSettingsProfileName, Settings, SettingsStore};
use std::sync::Arc;
use theme::{Appearance, SystemAppearance, Theme, ThemeMeta, ThemeRegistry};
use theme_settings::{
    ThemeAppearanceMode, ThemeName, ThemeSelection, ThemeSettings, appearance_to_mode,
};
use ui::{ListItem, ListItemSpacing, prelude::*, v_flex};
use util::ResultExt;
use workspace::{ModalView, Workspace, ui::HighlightedLabel, with_active_or_new_workspace};
use zed_actions::{ExtensionCategoryFilter, Extensions};

use crate::icon_theme_selector::{IconThemeSelector, IconThemeSelectorDelegate};

actions!(
    theme_selector,
    [
        /// Reloads all themes from disk.
        Reload
    ]
);

pub fn init(cx: &mut App) {
    cx.on_action(|action: &zed_actions::theme_selector::Toggle, cx| {
        let action = action.clone();
        with_active_or_new_workspace(cx, move |workspace, window, cx| {
            toggle_theme_selector(workspace, &action, window, cx);
        });
    });
    cx.on_action(|action: &zed_actions::icon_theme_selector::Toggle, cx| {
        let action = action.clone();
        with_active_or_new_workspace(cx, move |workspace, window, cx| {
            toggle_icon_theme_selector(workspace, &action, window, cx);
        });
    });
}

fn toggle_theme_selector(
    workspace: &mut Workspace,
    toggle: &zed_actions::theme_selector::Toggle,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let fs = workspace.app_state().fs.clone();
    workspace.toggle_modal(window, cx, |window, cx| {
        let delegate = ThemeSelectorDelegate::new(
            cx.entity().downgrade(),
            fs,
            toggle.themes_filter.as_ref(),
            cx,
        );
        ThemeSelector::new(delegate, window, cx)
    });
}

fn toggle_icon_theme_selector(
    workspace: &mut Workspace,
    toggle: &zed_actions::icon_theme_selector::Toggle,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let fs = workspace.app_state().fs.clone();
    workspace.toggle_modal(window, cx, |window, cx| {
        let delegate = IconThemeSelectorDelegate::new(
            cx.entity().downgrade(),
            fs,
            toggle.themes_filter.as_ref(),
            cx,
        );
        IconThemeSelector::new(delegate, window, cx)
    });
}

impl ModalView for ThemeSelector {
    fn on_before_dismiss(
        &mut self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> workspace::DismissDecision {
        self.picker.update(cx, |picker, cx| {
            picker.delegate.revert_theme(cx);
        });
        workspace::DismissDecision::Dismiss(true)
    }
}

struct ThemeSelector {
    picker: Entity<Picker<ThemeSelectorDelegate>>,
}

impl EventEmitter<DismissEvent> for ThemeSelector {}

impl Focusable for ThemeSelector {
    fn focus_handle(&self, cx: &App) -> gpui::FocusHandle {
        self.picker.focus_handle(cx)
    }
}

impl Render for ThemeSelector {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .key_context("ThemeSelector")
            .w(rems(34.))
            .child(self.picker.clone())
    }
}

impl ThemeSelector {
    pub fn new(
        delegate: ThemeSelectorDelegate,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let picker = cx.new(|cx| Picker::uniform_list(delegate, window, cx));
        Self { picker }
    }
}

struct ThemeSelectorDelegate {
    fs: Arc<dyn Fs>,
    themes: Vec<ThemeMeta>,
    matches: Vec<StringMatch>,
    /// The theme that was selected before the `ThemeSelector` menu was opened.
    ///
    /// We use this to return back to theme that was set if the user dismisses the menu.
    original_theme_settings: ThemeSettings,
    /// The current system appearance.
    original_system_appearance: Appearance,
    /// The id of the original theme in the list of themes.
    /// Using `Option<usize>` instead of `usize` because it's possible that the
    /// original theme is not present in the list of themes when it is first
    /// built, depending on the provided `themes_filter`. For example, when a
    /// theme is installed, the `themes_filter` is set to the new theme names
    /// and, if we used `unwrap_or(0)` as a fallback, the first theme in the
    /// list would be shown as "active".
    original_theme_id: Option<usize>,
    /// The currently selected new theme.
    new_theme: Arc<Theme>,
    selection_completed: bool,
    save_error: Option<SharedString>,
    selected_theme: Option<Arc<Theme>>,
    selected_index: usize,
    selector: WeakEntity<ThemeSelector>,
}

impl ThemeSelectorDelegate {
    fn new(
        selector: WeakEntity<ThemeSelector>,
        fs: Arc<dyn Fs>,
        themes_filter: Option<&Vec<String>>,
        cx: &mut Context<ThemeSelector>,
    ) -> Self {
        let original_theme = cx.theme().clone();
        let original_theme_settings = ThemeSettings::get_global(cx).clone();
        let original_system_appearance = SystemAppearance::global(cx).0;

        let registry = ThemeRegistry::global(cx);
        let mut themes = registry
            .list()
            .into_iter()
            .filter(|meta| {
                if let Some(theme_filter) = themes_filter {
                    theme_filter.contains(&meta.name.to_string())
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        // Sort by dark vs light, then by name.
        themes.sort_unstable_by(|a, b| {
            a.appearance
                .is_light()
                .cmp(&b.appearance.is_light())
                .then(a.name.cmp(&b.name))
        });

        let original_theme_id = themes
            .iter()
            .position(|meta| meta.name == original_theme.name);

        let matches: Vec<StringMatch> = themes
            .iter()
            .enumerate()
            .map(|(id, meta)| StringMatch {
                candidate_id: id,
                score: 0.0,
                positions: Default::default(),
                string: meta.name.to_string(),
            })
            .collect();

        // The current theme is likely in this list, so default to first showing that.
        let selected_index = matches
            .iter()
            .position(|mat| mat.string == original_theme.name)
            .unwrap_or(0);

        Self {
            fs,
            themes,
            matches,
            original_theme_settings,
            original_system_appearance,
            original_theme_id,
            new_theme: original_theme, // Start with the original theme.
            selected_index,
            selection_completed: false,
            save_error: None,
            selected_theme: None,
            selector,
        }
    }

    fn is_original_theme(&self, index: usize) -> bool {
        self.matches
            .get(index)
            .zip(self.original_theme_id)
            .is_some_and(|(mat, original_theme_id)| mat.candidate_id == original_theme_id)
    }

    fn show_selected_theme(
        &mut self,
        cx: &mut Context<Picker<ThemeSelectorDelegate>>,
    ) -> Option<Arc<Theme>> {
        if self.selection_completed {
            return None;
        }
        if let Some(mat) = self.matches.get(self.selected_index) {
            let registry = ThemeRegistry::global(cx);

            match registry.get(&mat.string) {
                Ok(theme) => {
                    self.set_theme(theme.clone(), cx);
                    Some(theme)
                }
                Err(error) => {
                    log::error!("error loading theme {}: {}", mat.string, error);
                    None
                }
            }
        } else {
            None
        }
    }

    fn revert_theme(&mut self, cx: &mut App) {
        if !self.selection_completed {
            SettingsStore::update_global(cx, |store, _| {
                store.restore_resolved_setting::<ThemeSettings>();
            });
            self.selection_completed = true;
        }
    }

    fn set_theme(&mut self, new_theme: Arc<Theme>, cx: &mut App) {
        // Update the global (in-memory) theme settings.
        SettingsStore::update_global(cx, |store, _| {
            override_global_theme(
                store,
                &new_theme,
                &self.original_theme_settings.theme,
                self.original_system_appearance,
            )
        });

        self.new_theme = new_theme;
    }
}

/// Overrides the global (in-memory) theme settings.
///
/// Note that this does **not** update the user's `settings.json` file (see the
/// [`ThemeSelectorDelegate::confirm`] method and [`theme_settings::set_theme`] function).
fn override_global_theme(
    store: &mut SettingsStore,
    new_theme: &Theme,
    original_theme: &ThemeSelection,
    system_appearance: Appearance,
) {
    let theme_name = ThemeName(new_theme.name.clone().into());
    let new_appearance = new_theme.appearance();
    let new_theme_is_light = new_appearance.is_light();

    let mut curr_theme_settings = store.get::<ThemeSettings>(None).clone();

    match (original_theme, &curr_theme_settings.theme) {
        // Override the currently selected static theme.
        (ThemeSelection::Static(_), ThemeSelection::Static(_)) => {
            curr_theme_settings.theme = ThemeSelection::Static(theme_name);
        }

        // If the current theme selection is dynamic, then only override the global setting for the
        // specific mode (light or dark).
        (
            ThemeSelection::Dynamic {
                mode: original_mode,
                light: original_light,
                dark: original_dark,
            },
            ThemeSelection::Dynamic { .. },
        ) => {
            let new_mode = update_mode_if_new_appearance_is_different_from_system(
                original_mode,
                system_appearance,
                new_appearance,
            );

            let updated_theme = retain_original_opposing_theme(
                new_theme_is_light,
                new_mode,
                theme_name,
                original_light,
                original_dark,
            );

            curr_theme_settings.theme = updated_theme;
        }

        // The theme selection mode changed while selecting new themes (someone edited the settings
        // file on disk while we had the dialogue open), so don't do anything.
        _ => return,
    };

    store.override_global(curr_theme_settings);
}

fn update_mode_if_new_appearance_is_different_from_system(
    _original_mode: &ThemeAppearanceMode,
    _system_appearance: Appearance,
    new_appearance: Appearance,
) -> ThemeAppearanceMode {
    // Picking an explicit theme must not revert on the next OS appearance change.
    appearance_to_mode(new_appearance)
}

/// Helper function for updating / displaying the [`ThemeSelection`] while using the theme selector.
///
/// We want to retain the alternate theme selection of the original settings (before the menu was
/// opened), not the currently selected theme (which likely has changed multiple times while the
/// menu has been open).
fn retain_original_opposing_theme(
    new_theme_is_light: bool,
    new_mode: ThemeAppearanceMode,
    theme_name: ThemeName,
    original_light: &ThemeName,
    original_dark: &ThemeName,
) -> ThemeSelection {
    if new_theme_is_light {
        ThemeSelection::Dynamic {
            mode: new_mode,
            light: theme_name,
            dark: original_dark.clone(),
        }
    } else {
        ThemeSelection::Dynamic {
            mode: new_mode,
            light: original_light.clone(),
            dark: theme_name,
        }
    }
}

fn theme_write_target<'a>(
    settings: &'a mut settings::UserSettingsContent,
    profile: Option<&str>,
) -> anyhow::Result<&'a mut settings::SettingsContent> {
    if let Some(profile) = profile {
        return settings
            .profiles
            .get_mut(profile)
            .map(|profile| profile.settings.as_mut())
            .ok_or_else(|| anyhow::anyhow!("The selected settings profile no longer exists"));
    }
    if let Some(platform) = settings
        .platform_overrides
        .get_mut_by_key(std::env::consts::OS)
        && platform.theme.theme.is_some()
    {
        return Ok(platform);
    }
    if let Some(channel) = settings
        .release_channel_overrides
        .get_mut_by_key(release_channel::RELEASE_CHANNEL.dev_name())
        && channel.theme.theme.is_some()
    {
        return Ok(channel);
    }
    Ok(&mut settings.content)
}

impl PickerDelegate for ThemeSelectorDelegate {
    type ListItem = ui::ListItem;

    fn name() -> &'static str {
        "theme selector"
    }

    fn placeholder_text(&self, _window: &mut Window, _cx: &mut App) -> Arc<str> {
        "Select Theme...".into()
    }

    fn match_count(&self) -> usize {
        self.matches.len()
    }

    fn confirm(
        &mut self,
        _secondary: bool,
        _window: &mut Window,
        cx: &mut Context<Picker<ThemeSelectorDelegate>>,
    ) {
        if self.selection_completed {
            return;
        }
        self.selection_completed = true;
        self.save_error = None;
        let theme_name: Arc<str> = self.new_theme.name.as_str().into();
        let profile = cx
            .try_global::<ActiveSettingsProfileName>()
            .map(|profile| profile.0.clone());
        telemetry::event!("Settings Changed", setting = "theme", value = theme_name);
        let completion = cx
            .global::<SettingsStore>()
            .update_user_settings_file_with_completion(self.fs.clone(), move |settings| {
                let target = theme_write_target(settings, profile.as_deref())?;
                target.theme.theme = Some(settings::ThemeSelection::Static(ThemeName(theme_name)));
                Ok(())
            });
        cx.spawn(async move |this, cx| {
            let result = completion.await.map_err(anyhow::Error::from).and_then(|result| result);
            this.update(cx, |picker, cx| {
                SettingsStore::update_global(cx, |store, _| store.restore_resolved_setting::<ThemeSettings>());
                match result {
                    Ok(()) => {
                        picker.delegate.selector.update(cx, |_, cx| cx.emit(DismissEvent)).log_err();
                    }
                    Err(error) => {
                        picker.delegate.selection_completed = false;
                        picker.delegate.save_error = Some(format!("Could not save theme: {error}. Fix settings file access or syntax, then select the theme to retry.").into());
                        cx.notify();
                    }
                }
            }).log_err();
        }).detach();
    }

    fn dismissed(&mut self, _: &mut Window, cx: &mut Context<Picker<ThemeSelectorDelegate>>) {
        self.revert_theme(cx);

        self.selector
            .update(cx, |_, cx| cx.emit(DismissEvent))
            .log_err();
    }

    fn selected_index(&self) -> usize {
        self.selected_index
    }

    fn set_selected_index(
        &mut self,
        ix: usize,
        _: &mut Window,
        cx: &mut Context<Picker<ThemeSelectorDelegate>>,
    ) {
        if self.selection_completed {
            return;
        }
        self.selected_index = ix;
        self.selected_theme = self.show_selected_theme(cx);
    }

    fn update_matches(
        &mut self,
        query: String,
        window: &mut Window,
        cx: &mut Context<Picker<ThemeSelectorDelegate>>,
    ) -> gpui::Task<()> {
        let background = cx.background_executor().clone();
        let candidates = self
            .themes
            .iter()
            .enumerate()
            .map(|(id, meta)| StringMatchCandidate::new(id, &meta.name))
            .collect::<Vec<_>>();

        cx.spawn_in(window, async move |this, cx| {
            let matches = if query.is_empty() {
                candidates
                    .into_iter()
                    .enumerate()
                    .map(|(index, candidate)| StringMatch {
                        candidate_id: index,
                        string: candidate.string,
                        positions: Vec::new(),
                        score: 0.0,
                    })
                    .collect()
            } else {
                match_strings(
                    &candidates,
                    &query,
                    false,
                    true,
                    100,
                    &Default::default(),
                    background,
                )
                .await
            };

            this.update(cx, |this, cx| {
                this.delegate.matches = matches;
                if query.is_empty() && this.delegate.selected_theme.is_none() {
                    this.delegate.selected_index = this
                        .delegate
                        .selected_index
                        .min(this.delegate.matches.len().saturating_sub(1));
                } else if let Some(selected) = this.delegate.selected_theme.as_ref() {
                    this.delegate.selected_index = this
                        .delegate
                        .matches
                        .iter()
                        .enumerate()
                        .find(|(_, mtch)| mtch.string == selected.name)
                        .map(|(ix, _)| ix)
                        .unwrap_or_default();
                } else {
                    this.delegate.selected_index = 0;
                }
                // Preserve the previously selected theme when the filter yields no results.
                if let Some(theme) = this.delegate.show_selected_theme(cx) {
                    this.delegate.selected_theme = Some(theme);
                }
            })
            .log_err();
        })
    }

    fn render_match(
        &self,
        ix: usize,
        selected: bool,
        _window: &mut Window,
        _cx: &mut Context<Picker<Self>>,
    ) -> Option<Self::ListItem> {
        let theme_match = &self.matches.get(ix)?;
        let is_original_theme = self.is_original_theme(ix);

        Some(
            ListItem::new(ix)
                .inset(true)
                .spacing(ListItemSpacing::Sparse)
                .toggle_state(selected)
                .child(HighlightedLabel::new(
                    theme_match.string.clone(),
                    theme_match.positions.clone(),
                ))
                .when(is_original_theme, |this| {
                    this.end_slot(Icon::new(IconName::Check).color(Color::Muted))
                }),
        )
    }

    fn render_footer(
        &self,
        _: &mut Window,
        cx: &mut Context<Picker<Self>>,
    ) -> Option<gpui::AnyElement> {
        Some(
            h_flex()
                .flex_wrap()
                .when_some(self.save_error.clone(), |this, error| {
                    this.child(
                        div()
                            .w_full()
                            .child(Label::new(error).size(LabelSize::Small).color(Color::Error)),
                    )
                })
                .p_2()
                .w_full()
                .justify_between()
                .gap_2()
                .border_t_1()
                .border_color(cx.theme().colors().border_variant)
                .child(
                    Button::new("docs", "View Theme Docs")
                        .end_icon(
                            Icon::new(IconName::ArrowUpRight)
                                .size(IconSize::Small)
                                .color(Color::Muted),
                        )
                        .on_click(cx.listener(|_, _, _, cx| {
                            cx.open_url("https://zed.dev/docs/themes");
                        })),
                )
                .child(
                    Button::new("more-themes", "Install Themes").on_click(cx.listener({
                        move |_, _, window, cx| {
                            window.dispatch_action(
                                Box::new(Extensions {
                                    category_filter: Some(ExtensionCategoryFilter::Themes),
                                    id: None,
                                }),
                                cx,
                            );
                        }
                    })),
                )
                .into_any_element(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{TestAppContext, VisualTestContext};
    use project::Project;
    use serde_json::json;
    use theme::{Appearance, ThemeFamily, ThemeRegistry, default_color_scales};
    use util::path;
    use workspace::MultiWorkspace;

    fn init_test(cx: &mut TestAppContext) -> Arc<workspace::AppState> {
        cx.update(|cx| {
            let app_state = workspace::AppState::test(cx);
            settings::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
            super::init(cx);
            app_state
        })
    }

    fn register_test_themes(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let registry = ThemeRegistry::global(cx);
            let base_theme = registry.get("One Dark").unwrap();

            let mut test_light = (*base_theme).clone();
            test_light.id = "test-light".to_string();
            test_light.name = "Test Light".into();
            test_light.appearance = Appearance::Light;

            let mut test_dark_a = (*base_theme).clone();
            test_dark_a.id = "test-dark-a".to_string();
            test_dark_a.name = "Test Dark A".into();

            let mut test_dark_b = (*base_theme).clone();
            test_dark_b.id = "test-dark-b".to_string();
            test_dark_b.name = "Test Dark B".into();

            registry.register_test_themes([ThemeFamily {
                id: "test-family".to_string(),
                name: "Test Family".into(),
                author: "test".into(),
                themes: vec![test_light, test_dark_a, test_dark_b],
                scales: default_color_scales(),
            }]);
        });
    }

    async fn setup_test(cx: &mut TestAppContext) -> Arc<workspace::AppState> {
        let app_state = init_test(cx);
        register_test_themes(cx);
        app_state
            .fs
            .as_fake()
            .insert_tree(path!("/test"), json!({}))
            .await;
        app_state
    }

    fn open_theme_selector(
        workspace: &Entity<workspace::Workspace>,
        cx: &mut VisualTestContext,
    ) -> Entity<Picker<ThemeSelectorDelegate>> {
        cx.dispatch_action(zed_actions::theme_selector::Toggle {
            themes_filter: None,
        });
        cx.run_until_parked();
        workspace.update(cx, |workspace, cx| {
            workspace
                .active_modal::<ThemeSelector>(cx)
                .expect("theme selector should be open")
                .read(cx)
                .picker
                .clone()
        })
    }

    fn selected_theme_name(
        picker: &Entity<Picker<ThemeSelectorDelegate>>,
        cx: &mut VisualTestContext,
    ) -> String {
        picker.read_with(cx, |picker, _| {
            picker
                .delegate
                .matches
                .get(picker.delegate.selected_index)
                .expect("selected index should point to a match")
                .string
                .clone()
        })
    }

    fn previewed_theme_name(
        picker: &Entity<Picker<ThemeSelectorDelegate>>,
        cx: &mut VisualTestContext,
    ) -> String {
        picker.read_with(cx, |picker, _| picker.delegate.new_theme.name.to_string())
    }

    #[gpui::test]
    async fn test_theme_selector_preserves_selection_on_empty_filter(cx: &mut TestAppContext) {
        let app_state = setup_test(cx).await;
        let project = Project::test(app_state.fs.clone(), [path!("/test").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace =
            multi_workspace.read_with(cx, |multi_workspace, _| multi_workspace.workspace().clone());
        let picker = open_theme_selector(&workspace, cx);

        let target_index = picker.read_with(cx, |picker, _| {
            picker
                .delegate
                .matches
                .iter()
                .position(|m| m.string == "Test Light")
                .unwrap()
        });
        picker.update_in(cx, |picker, window, cx| {
            picker.set_selected_index(target_index, None, true, window, cx);
        });
        cx.run_until_parked();

        assert_eq!(previewed_theme_name(&picker, cx), "Test Light");

        picker.update_in(cx, |picker, window, cx| {
            picker.update_matches("zzz".to_string(), window, cx);
        });
        cx.run_until_parked();

        picker.update_in(cx, |picker, window, cx| {
            picker.update_matches("".to_string(), window, cx);
        });
        cx.run_until_parked();

        assert_eq!(
            selected_theme_name(&picker, cx),
            "Test Light",
            "selected theme should be preserved after clearing an empty filter"
        );
        assert_eq!(
            previewed_theme_name(&picker, cx),
            "Test Light",
            "previewed theme should be preserved after clearing an empty filter"
        );
    }
    #[gpui::test]
    async fn confirmed_theme_survives_profile_resolution(cx: &mut TestAppContext) {
        let app_state = setup_test(cx).await;
        let contents = r#"{"theme":"Test Dark A","ui_font_size":18,"profiles":{"work":{"settings":{"theme":"Test Dark B","unstable.ui_density":"compact"}}}}"#;
        app_state
            .fs
            .create_dir(paths::settings_file().parent().expect("settings directory"))
            .await
            .expect("isolated settings directory");
        app_state
            .fs
            .as_fake()
            .insert_file(paths::settings_file(), contents.as_bytes().to_vec())
            .await;
        cx.update(|cx| {
            cx.set_global(settings::ActiveSettingsProfileName("work".into()));
            SettingsStore::update_global(cx, |store, cx| {
                store.set_user_settings(contents, cx).unwrap();
            });
        });
        let project = Project::test(app_state.fs.clone(), [path!("/test").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |view, _| view.workspace().clone());
        let picker = open_theme_selector(&workspace, cx);
        picker.update_in(cx, |picker, window, cx| {
            let index = picker
                .delegate
                .matches
                .iter()
                .position(|item| item.string == "Test Light")
                .expect("theme registered");
            picker.set_selected_index(index, None, true, window, cx);
            picker.delegate.confirm(false, window, cx);
        });
        cx.run_until_parked();
        let saved = app_state
            .fs
            .load(paths::settings_file())
            .await
            .expect("saved settings");
        let value: serde_json::Value = serde_json::from_str(&saved).expect("valid settings");
        assert_eq!(value["profiles"]["work"]["settings"]["theme"], "Test Light");
        assert_eq!(value["theme"], "Test Dark A");
        assert_eq!(value["ui_font_size"], 18);
        assert_eq!(
            value["profiles"]["work"]["settings"]["unstable.ui_density"],
            "compact"
        );
        for appearance in [Appearance::Dark, Appearance::Light] {
            cx.update(|_, cx| {
                *SystemAppearance::global_mut(cx) = SystemAppearance(appearance);
                theme_settings::reload_theme(cx);
                assert_eq!(cx.theme().name.as_ref(), "Test Light");
                assert_eq!(
                    ThemeSettings::get_global(cx).ui_density,
                    theme::UiDensity::Compact
                );
            });
        }
        let reopened = open_theme_selector(&workspace, cx);
        reopened.update_in(cx, |picker, window, cx| {
            picker.delegate.dismissed(window, cx)
        });
        cx.run_until_parked();
        cx.update(|_, cx| {
            let mut restored = SettingsStore::test(cx);
            restored.set_user_settings(&saved, cx).unwrap();
            cx.set_global(restored);
            theme_settings::reload_theme(cx);
            assert_eq!(cx.theme().name.as_ref(), "Test Light");
        });

        cx.update(|_, cx| {
            SettingsStore::update_global(cx, |store, _| {
                store.restore_resolved_setting::<ThemeSettings>()
            });
            assert_eq!(
                ThemeSettings::get_global(cx)
                    .theme
                    .name(Appearance::Dark)
                    .0
                    .as_ref(),
                "Test Light"
            );
        });
    }
    #[gpui::test]
    async fn failed_theme_write_is_visible_and_retryable(cx: &mut TestAppContext) {
        let app_state = setup_test(cx).await;
        app_state
            .fs
            .create_dir(paths::settings_file().parent().expect("settings parent"))
            .await
            .expect("settings directory");
        app_state
            .fs
            .as_fake()
            .insert_file(paths::settings_file(), b"{broken".to_vec())
            .await;
        let project = Project::test(app_state.fs.clone(), [path!("/test").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |view, _| view.workspace().clone());
        let picker = open_theme_selector(&workspace, cx);
        picker.update_in(cx, |picker, window, cx| {
            let index = picker
                .delegate
                .matches
                .iter()
                .position(|item| item.string == "Test Light")
                .expect("theme registered");
            picker.set_selected_index(index, None, true, window, cx);
            picker.delegate.confirm(false, window, cx);
        });
        cx.run_until_parked();
        picker.read_with(cx, |picker, _| {
            assert!(
                picker
                    .delegate
                    .save_error
                    .as_ref()
                    .expect("actionable error")
                    .contains("Fix settings")
            );
            assert!(!picker.delegate.selection_completed);
        });
        assert_eq!(
            app_state
                .fs
                .load(paths::settings_file())
                .await
                .expect("unchanged file"),
            "{broken"
        );
        app_state
            .fs
            .as_fake()
            .insert_file(paths::settings_file(), b"{}".to_vec())
            .await;
        picker.update_in(cx, |picker, window, cx| {
            picker.delegate.confirm(false, window, cx)
        });
        cx.run_until_parked();
        cx.update(|_, cx| assert_eq!(cx.theme().name.as_ref(), "Test Light"));
        picker.read_with(cx, |picker, _| {
            assert!(picker.delegate.save_error.is_none())
        });
    }

    #[gpui::test(iterations = 5)]
    async fn queued_theme_writes_keep_latest_and_preserve_other_preferences(
        cx: &mut TestAppContext,
    ) {
        let app_state = setup_test(cx).await;
        app_state
            .fs
            .create_dir(paths::settings_file().parent().expect("settings parent"))
            .await
            .expect("settings directory");
        let contents = r#"{"theme":"Test Dark A","ui_font_size":19,"icon_theme":"Zed (Default)","unstable.ui_density":"compact"}"#;
        app_state
            .fs
            .as_fake()
            .insert_file(paths::settings_file(), contents.as_bytes().to_vec())
            .await;
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.watch_settings_files(app_state.fs.clone(), cx, |_, _, _| {});
            });
        });
        app_state.fs.as_fake().pause_events();
        let writes = cx.update(|cx| {
            ["Test Light", "Test Dark B"].map(|name| {
                cx.global::<SettingsStore>()
                    .update_user_settings_file_with_completion(
                        app_state.fs.clone(),
                        move |content| {
                            theme_write_target(content, None)?.theme.theme =
                                Some(settings::ThemeSelection::Static(ThemeName(name.into())));
                            Ok(())
                        },
                    )
            })
        });
        for write in writes {
            write
                .await
                .expect("queue completes")
                .expect("write succeeds");
        }
        app_state.fs.as_fake().flush_events(100);
        cx.run_until_parked();
        let saved = app_state
            .fs
            .load(paths::settings_file())
            .await
            .expect("saved settings");
        let value: serde_json::Value = serde_json::from_str(&saved).expect("valid settings");
        assert_eq!(value["theme"], "Test Dark B");
        assert_eq!(value["ui_font_size"], 19);
        assert_eq!(value["icon_theme"], "Zed (Default)");
        assert_eq!(value["unstable.ui_density"], "compact");
        cx.update(|cx| assert_eq!(cx.theme().name.as_ref(), "Test Dark B"));
    }

    #[test]
    fn theme_write_uses_applicable_override_and_preserves_system_configuration() {
        let mut content: settings::UserSettingsContent = serde_json::from_value(json!({
            "theme":"Test Dark A",
            std::env::consts::OS: {"theme":{"mode":"system","light":"Test Light","dark":"Test Dark B"},"ui_font_size":20}
        })).expect("settings");
        theme_write_target(&mut content, None)
            .expect("applicable layer")
            .theme
            .theme = Some(settings::ThemeSelection::Static(ThemeName(
            "Test Light".into(),
        )));
        assert_eq!(
            content.content.theme.theme,
            Some(settings::ThemeSelection::Static(ThemeName(
                "Test Dark A".into()
            )))
        );
        let platform = content
            .platform_overrides
            .get_by_key(std::env::consts::OS)
            .expect("platform");
        assert_eq!(
            platform.theme.theme,
            Some(settings::ThemeSelection::Static(ThemeName(
                "Test Light".into()
            )))
        );
        assert_eq!(platform.theme.ui_font_size, Some(20.0.into()));
        let dynamic: settings::ThemeSelection = serde_json::from_value(
            json!({"mode":"system","light":"Test Light","dark":"Test Dark B"}),
        )
        .expect("system selection");
        assert!(matches!(
            dynamic,
            settings::ThemeSelection::Dynamic {
                mode: ThemeAppearanceMode::System,
                ..
            }
        ));
    }

    #[gpui::test]
    async fn saved_system_mode_follows_appearance_and_cancel_preserves_preferences(
        cx: &mut TestAppContext,
    ) {
        let app_state = setup_test(cx).await;
        let contents = r#"{"theme":{"mode":"system","light":"Test Light","dark":"Test Dark B"},"ui_font_size":19,"unstable.ui_density":"compact"}"#;
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.set_user_settings(contents, cx).unwrap()
            })
        });
        let project = Project::test(app_state.fs.clone(), [path!("/test").as_ref()], cx).await;
        let (multi_workspace, cx) =
            cx.add_window_view(|window, cx| MultiWorkspace::test_new(project.clone(), window, cx));
        let workspace = multi_workspace.read_with(cx, |view, _| view.workspace().clone());
        let picker = open_theme_selector(&workspace, cx);
        picker.update_in(cx, |picker, window, cx| {
            picker.delegate.dismissed(window, cx)
        });
        for (appearance, name) in [
            (Appearance::Light, "Test Light"),
            (Appearance::Dark, "Test Dark B"),
        ] {
            cx.update(|_, cx| {
                *SystemAppearance::global_mut(cx) = SystemAppearance(appearance);
                theme_settings::reload_theme(cx);
                assert_eq!(cx.theme().name.as_ref(), name);
                assert_eq!(
                    ThemeSettings::get_global(cx).ui_density,
                    theme::UiDensity::Compact
                );
            });
        }
    }
}
