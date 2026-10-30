use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context as _, Result, bail};
use editor::Editor;
use feature_flags::{
    FeatureFlag as _, FeatureFlagAppExt as _, RustNamedCargoPresetsExperimentalFeatureFlag,
};
use gpui::{
    App, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable, Render, WeakEntity,
    Window, actions,
};
use settings::{
    CargoPresetSettingsContent, CargoSettingsContent, update_settings_file_with_completion,
};
use ui::{Button, ButtonStyle, Modal, ModalFooter, ModalHeader, Section, prelude::*};
use workspace::ModalView;

use crate::{
    CargoAction, CargoCompileContext, CargoPanel, CargoPreset, CargoPresetScope,
    CargoTargetSelector, CargoWorkingDirectoryPolicy, compile_preset, parse_preset_content,
};

actions!(
    cargo_preset_editor,
    [
        RunDraft,
        DebugDraft,
        ProfileDraft,
        SaveDraftForUser,
        SaveDraftForProject
    ]
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoPresetSaveScope {
    User,
    Project,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CargoPresetValidationContext {
    pub packages: BTreeSet<String>,
    pub profiles: BTreeSet<String>,
    pub targets: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CargoPresetPreview {
    pub program: String,
    pub arguments: Vec<String>,
    pub environment_keys: Vec<String>,
}

impl CargoPresetPreview {
    pub fn display(&self) -> String {
        let mut tokens = Vec::with_capacity(self.arguments.len() + 1);
        tokens.push(self.program.clone());
        tokens.extend(self.arguments.iter().cloned());
        let environment = if self.environment_keys.is_empty() {
            String::new()
        } else {
            format!("; environment keys: {}", self.environment_keys.join(", "))
        };
        format!("{}{}", tokens.join(" "), environment)
    }
}

pub fn validate_preset_draft(
    preset: &CargoPreset,
    context: &CargoPresetValidationContext,
) -> Result<()> {
    if preset.scope == CargoPresetScope::Package {
        let package = preset
            .package
            .as_deref()
            .context("Package scope requires a package")?;
        if !context.packages.is_empty() && !context.packages.contains(package) {
            bail!("Package `{package}` is no longer present in the Cargo snapshot");
        }
    }
    if let Some(profile) = preset.profile.as_deref()
        && !context.profiles.is_empty()
        && !context.profiles.contains(profile)
    {
        bail!("Profile `{profile}` is not present in the Cargo snapshot");
    }
    let target_name = match preset.target.as_ref() {
        Some(
            CargoTargetSelector::Binary(name)
            | CargoTargetSelector::Example(name)
            | CargoTargetSelector::Test(name)
            | CargoTargetSelector::Bench(name),
        ) => Some(name),
        _ => None,
    };
    if let Some(target_name) = target_name
        && !context.targets.is_empty()
        && !context.targets.contains(target_name)
    {
        bail!("Target `{target_name}` is no longer present in the Cargo snapshot");
    }
    if preset
        .features
        .iter()
        .any(|feature| feature.trim().is_empty())
    {
        bail!("Feature names must not be empty");
    }
    if preset
        .args
        .iter()
        .chain(&preset.trailing_args)
        .any(|argument| argument.contains('\0'))
    {
        bail!("Cargo arguments must not contain NUL bytes");
    }
    if let CargoWorkingDirectoryPolicy::Custom(path) = &preset.working_directory
        && (path.trim().is_empty() || path.contains('\0'))
    {
        bail!("Custom working directory must be non-empty and contain no NUL bytes");
    }
    Ok(())
}

pub fn redacted_preset_preview(preset: &CargoPreset) -> Result<CargoPresetPreview> {
    if let Some(native) = &preset.native_profiler {
        crate::CargoNativeProfiler::from_settings(&native.settings_content())?;
        return Ok(CargoPresetPreview {
            program: preset.profile_action_label().into(),
            arguments: vec!["selected Cargo Build then Samply 0.13.1".into()],
            environment_keys: preset.environment_keys(),
        });
    }
    let compiled = compile_preset(
        preset,
        &CargoCompileContext {
            package_name: preset.package.clone(),
            ..CargoCompileContext::default()
        },
        None,
    )?;
    Ok(CargoPresetPreview {
        program: compiled.task_template.command,
        arguments: compiled.task_template.args,
        environment_keys: preset.environment_keys(),
    })
}

pub fn preset_settings_content(
    preset: &CargoPreset,
    scope: CargoPresetSaveScope,
) -> CargoPresetSettingsContent {
    let (target_kind, target_name) = match &preset.target {
        Some(CargoTargetSelector::Library) => (Some("lib".to_string()), None),
        Some(CargoTargetSelector::Binary(name)) => (Some("bin".to_string()), Some(name.clone())),
        Some(CargoTargetSelector::Example(name)) => {
            (Some("example".to_string()), Some(name.clone()))
        }
        Some(CargoTargetSelector::Test(name)) => (Some("test".to_string()), Some(name.clone())),
        Some(CargoTargetSelector::Bench(name)) => (Some("bench".to_string()), Some(name.clone())),
        Some(CargoTargetSelector::AllTargets) => (Some("all_targets".to_string()), None),
        None => (None, None),
    };
    let (working_directory, custom_working_directory) = match &preset.working_directory {
        CargoWorkingDirectoryPolicy::Context => ("context", None),
        CargoWorkingDirectoryPolicy::Workspace => ("workspace", None),
        CargoWorkingDirectoryPolicy::Package => ("package", None),
        CargoWorkingDirectoryPolicy::Custom(path) => ("custom", Some(path.clone())),
    };
    CargoPresetSettingsContent {
        label: Some(preset.label.clone()),
        subcommand: Some(preset.subcommand.as_str().to_string()),
        scope: Some(
            match preset.scope {
                CargoPresetScope::Workspace => "workspace",
                CargoPresetScope::Package => "package",
            }
            .to_string(),
        ),
        package: preset.package.clone(),
        target_kind,
        target_name,
        profile: preset.profile.clone(),
        profiler: preset
            .profiler
            .as_ref()
            .map(crate::cargo_profile::CargoProfiler::settings_content),
        native_profiler: preset
            .native_profiler
            .as_ref()
            .map(crate::cargo_profile::CargoNativeProfiler::settings_content),
        features: Some(preset.features.clone()),
        default_features: preset.default_features,
        target_triple: preset.target_triple.clone(),
        toolchain: preset.toolchain.clone(),
        pre_launch_task: preset.pre_launch_task.clone(),
        args: Some(preset.args.clone()),
        trailing_args: Some(preset.trailing_args.clone()),
        environment: match scope {
            CargoPresetSaveScope::User => Some(preset.environment.clone()),
            CargoPresetSaveScope::Project => None,
        },
        working_directory: Some(working_directory.to_string()),
        custom_working_directory,
        reveal: Some(
            match preset.presentation.reveal {
                task::RevealStrategy::Always => "always",
                task::RevealStrategy::NoFocus => "no_focus",
                task::RevealStrategy::Never => "never",
            }
            .to_string(),
        ),
        reveal_target: Some(
            match preset.presentation.reveal_target {
                task::RevealTarget::Dock => "dock",
                task::RevealTarget::Center => "center",
            }
            .to_string(),
        ),
        hide: Some(
            match preset.presentation.hide {
                task::HideStrategy::Never => "never",
                task::HideStrategy::Always => "always",
                task::HideStrategy::OnSuccess => "on_success",
            }
            .to_string(),
        ),
        save: Some(
            match preset.presentation.save {
                task::SaveStrategy::None => "none",
                task::SaveStrategy::Current => "current",
                task::SaveStrategy::All => "all",
            }
            .to_string(),
        ),
        use_new_terminal: Some(preset.presentation.use_new_terminal),
        allow_concurrent_runs: Some(preset.presentation.allow_concurrent_runs),
    }
}

pub fn apply_preset_save(
    settings: &mut CargoSettingsContent,
    preset: &CargoPreset,
    scope: CargoPresetSaveScope,
    trusted: bool,
    confirmed: bool,
) -> Result<()> {
    if scope == CargoPresetSaveScope::Project {
        if !trusted {
            bail!("Trust this worktree before saving a shared Cargo preset");
        }
        if !confirmed {
            bail!(
                "Confirm that the preset will be shared in .zed/settings.json; environment values are omitted"
            );
        }
    }
    settings.schema_version = Some(crate::CARGO_PRESET_SCHEMA_VERSION);
    settings
        .presets
        .insert(preset.id.clone(), preset_settings_content(preset, scope));
    Ok(())
}

pub(crate) fn named_preset_save_revocation(cx: &mut App) -> (Arc<AtomicBool>, gpui::Subscription) {
    let revoked = Arc::new(AtomicBool::new(false));
    let subscription = cx.observe_global::<feature_flags::FeatureFlagStore>({
        let revoked = revoked.clone();
        move |cx| {
            if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
                revoked.store(true, Ordering::SeqCst);
            }
        }
    });
    (revoked, subscription)
}

pub struct CargoPresetEditor {
    panel: Option<WeakEntity<CargoPanel>>,
    editor: Entity<Editor>,
    preset_id: String,
    validation_context: CargoPresetValidationContext,
    project_save_confirmation: bool,
    last_error: Option<String>,
}

impl CargoPresetEditor {
    pub fn new(
        panel: WeakEntity<CargoPanel>,
        preset: CargoPreset,
        validation_context: CargoPresetValidationContext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new_inner(Some(panel), preset, validation_context, window, cx)
    }

    fn new_inner(
        panel: Option<WeakEntity<CargoPanel>>,
        preset: CargoPreset,
        validation_context: CargoPresetValidationContext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        RustNamedCargoPresetsExperimentalFeatureFlag::watch(cx);
        feature_flags::RustNativeProfilingExperimentalFeatureFlag::watch(cx);
        let content = preset_settings_content(&preset, CargoPresetSaveScope::User);
        let initial_text = serde_json::to_string_pretty(&content)
            .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"));
        let editor = cx.new(|cx| {
            let mut editor = Editor::auto_height(8, 24, window, cx);
            editor.set_text(initial_text, window, cx);
            editor
        });
        Self {
            panel,
            editor,
            preset_id: preset.id,
            validation_context,
            project_save_confirmation: false,
            last_error: None,
        }
    }

    fn draft(&self, cx: &App) -> Result<CargoPreset> {
        anyhow::ensure!(
            cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
            "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in"
        );
        let text = self.editor.read(cx).text(cx);
        let content: CargoPresetSettingsContent =
            serde_json::from_str(&text).context("Preset draft is not valid JSON")?;
        let preset = parse_preset_content(&self.preset_id, &content)?;
        validate_preset_draft(&preset, &self.validation_context)?;
        Ok(preset)
    }

    fn run_action(&mut self, action: CargoAction, window: &mut Window, cx: &mut Context<Self>) {
        match self.draft(cx).and_then(|preset| {
            crate::ensure_experimental_cargo_action(action, Some(&preset), cx)?;
            Ok(preset)
        }) {
            Ok(preset) => match self.panel.as_ref().map(|panel| {
                panel.update(cx, |panel, cx| {
                    panel.execute_selected_action_with_preset(action, Some(preset), window, cx)
                })
            }) {
                Some(Ok(())) => cx.emit(DismissEvent),
                Some(Err(error)) => {
                    self.last_error = Some(format!("Cargo panel is unavailable: {error}"));
                    cx.notify();
                }
                None => {
                    self.last_error = Some("Cargo panel is unavailable".to_string());
                    cx.notify();
                }
            },
            Err(error) => {
                self.last_error = Some(error.to_string());
                cx.notify();
            }
        }
    }

    fn save_user(&mut self, cx: &mut Context<Self>) {
        let preset = match self.draft(cx) {
            Ok(preset) => preset,
            Err(error) => {
                self.last_error = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        let (revoked, experimental_subscription) = named_preset_save_revocation(cx);
        let mutation_revoked = revoked.clone();
        let (sender, mutation) = futures::channel::oneshot::channel();
        let completion = update_settings_file_with_completion(
            <dyn fs::Fs>::global(cx),
            cx,
            move |settings, cx| {
                let result = (|| {
                    anyhow::ensure!(
                        !mutation_revoked.load(Ordering::SeqCst)
                            && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
                        "Named Cargo preset opt-in was revoked before saving"
                    );
                    apply_preset_save(
                        settings.project.cargo.get_or_insert_default(),
                        &preset,
                        CargoPresetSaveScope::User,
                        true,
                        true,
                    )
                })();
                if sender.send(result).is_err() {
                    log::debug!("Cargo preset editor was dropped before save completion");
                }
            },
        );
        cx.spawn(async move |this, cx| {
            let result: Result<()> = async {
                completion
                    .await
                    .context("Cargo settings save completion was cancelled")??;
                mutation
                    .await
                    .context("Cargo preset update was cancelled")??;
                anyhow::ensure!(
                    !revoked.load(Ordering::SeqCst),
                    "Named Cargo preset opt-in was revoked while saving; settings may already have been written"
                );
                Ok(())
            }
            .await;
            drop(experimental_subscription);
            if let Err(error) = this.update(cx, |this, cx| match result {
                Ok(()) => {
                    this.last_error = None;
                    cx.emit(DismissEvent);
                }
                Err(error) => {
                    this.last_error = Some(error.to_string());
                    cx.notify();
                }
            }) {
                log::debug!("Cargo preset editor was dropped: {error:#}");
            }
        })
        .detach();
    }

    fn save_project(&mut self, cx: &mut Context<Self>) {
        if !cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>() {
            self.last_error = Some(
                "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in"
                    .into(),
            );
            cx.notify();
            return;
        }
        if !self.project_save_confirmation {
            self.project_save_confirmation = true;
            self.last_error = Some(
                "Save to .zed/settings.json? This preset is shared with the repository; environment values will be omitted. Activate Save for Project again to confirm."
                    .to_string(),
            );
            cx.notify();
            return;
        }
        match self.draft(cx) {
            Ok(preset) => {
                match self.panel.as_ref().map(|panel| {
                    panel.update(cx, |panel, cx| panel.save_project_preset(preset, cx))
                }) {
                    Some(Ok(Ok(()))) => cx.emit(DismissEvent),
                    Some(Ok(Err(error))) => {
                        self.last_error = Some(error.to_string());
                        cx.notify();
                    }
                    Some(Err(error)) => {
                        self.last_error = Some(format!("Cargo panel is unavailable: {error}"));
                        cx.notify();
                    }
                    None => {
                        self.last_error = Some("Cargo panel is unavailable".to_string());
                        cx.notify();
                    }
                }
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                cx.notify();
            }
        }
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }
}

impl EventEmitter<DismissEvent> for CargoPresetEditor {}
impl ModalView for CargoPresetEditor {}

impl Focusable for CargoPresetEditor {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.editor.focus_handle(cx)
    }
}

impl Render for CargoPresetEditor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let draft = self.draft(cx);
        let enabled = cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>();
        let profile_label = draft
            .as_ref()
            .map(|preset| preset.profile_action_label())
            .unwrap_or("Profile");
        let profile_enabled = draft.as_ref().is_ok_and(|preset| {
            crate::ensure_experimental_cargo_action(CargoAction::Profile, Some(preset), cx).is_ok()
        });
        let preview = draft
            .and_then(|preset| redacted_preset_preview(&preset))
            .map(|preview| preview.display())
            .unwrap_or_else(|error| format!("Draft unavailable: {error}"));
        v_flex()
            .id("cargo-preset-editor")
            .key_context("CargoPresetEditor")
            .track_focus(&self.focus_handle(cx))
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(|this, _: &RunDraft, window, cx| {
                this.run_action(CargoAction::Run, window, cx)
            }))
            .on_action(cx.listener(|this, _: &DebugDraft, window, cx| {
                this.run_action(CargoAction::Debug, window, cx)
            }))
            .on_action(cx.listener(|this, _: &ProfileDraft, window, cx| {
                this.run_action(CargoAction::Profile, window, cx)
            }))
            .on_action(cx.listener(|this, _: &SaveDraftForUser, _, cx| this.save_user(cx)))
            .on_action(cx.listener(|this, _: &SaveDraftForProject, _, cx| this.save_project(cx)))
            .w_128()
            .elevation_3(cx)
            .child(
                Modal::new("cargo-preset-editor-modal", None)
                    .header(
                        ModalHeader::new()
                            .headline("Edit Cargo preset")
                            .show_dismiss_button(true),
                    )
                    .section(
                        Section::new()
                            .child(self.editor.clone())
                            .child(div().mt_2().text_sm().child(
                                "For macOS native profiling, set native_profiler to an object with collector_path for user-installed Samply 0.13.1 and a project-relative profile_path ending in .json. Set profiler to null. Binary and example targets currently use the host default dev Build; other native selections and custom Build options remain incomplete."
                            ))
                            .child(div().mt_2().text_sm().child(format!("Preview: {preview}")))
                            .when_some(self.last_error.clone(), |section, error| {
                                section.child(div().mt_2().text_sm().child(error))
                            }),
                    )
                    .footer(
                        ModalFooter::new().end_slot(
                            h_flex()
                                .gap_2()
                                .child(
                                    Button::new("cargo-preset-save-project", "Save for Project")
                                        .style(ButtonStyle::Subtle)
                                        .disabled(!enabled)
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.save_project(cx)),
                                        ),
                                )
                                .child(
                                    Button::new("cargo-preset-save-user", "Save for User")
                                        .style(ButtonStyle::Subtle)
                                        .disabled(!enabled)
                                        .on_click(cx.listener(|this, _, _, cx| this.save_user(cx))),
                                )
                                .child(
                                    Button::new("cargo-preset-debug", "Debug")
                                        .style(ButtonStyle::Subtle)
                                        .disabled(!enabled)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.run_action(CargoAction::Debug, window, cx)
                                        })),
                                )
                                .child(
                                    Button::new("cargo-preset-profile", profile_label)
                                        .style(ButtonStyle::Subtle)
                                        .disabled(!profile_enabled)
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.run_action(CargoAction::Profile, window, cx)
                                        })),
                                )
                                .child(Button::new("cargo-preset-run", "Run").disabled(!enabled).on_click(
                                    cx.listener(|this, _, window, cx| {
                                        this.run_action(CargoAction::Run, window, cx)
                                    }),
                                )),
                        ),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use collections::HashMap;
    use gpui::TestAppContext;
    use settings::CargoSettingsContent;

    use super::*;
    use crate::{CargoSubcommand, parse_settings_content};

    #[test]
    fn cargo_preset_editor_validates_and_redacts_environment_values() {
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.id = "draft".to_string();
        preset.scope = CargoPresetScope::Package;
        preset.package = Some("member".to_string());
        preset.environment = HashMap::from_iter([
            ("TOKEN".to_string(), "top-secret".to_string()),
            ("RUSTFLAGS".to_string(), "-C debuginfo=1".to_string()),
        ]);
        let context = CargoPresetValidationContext {
            packages: BTreeSet::from_iter(["member".to_string()]),
            ..CargoPresetValidationContext::default()
        };
        validate_preset_draft(&preset, &context).expect("current draft should validate");
        let preview = redacted_preset_preview(&preset)
            .expect("valid draft should preview")
            .display();
        assert!(preview.contains("RUSTFLAGS, TOKEN"));
        assert!(!preview.contains("top-secret"));
        assert!(!preview.contains("debuginfo"));

        preset.package = Some("removed".to_string());
        assert!(
            validate_preset_draft(&preset, &context)
                .expect_err("stale package should fail")
                .to_string()
                .contains("no longer present")
        );
    }

    #[test]
    fn cargo_preset_save_scope_requires_trust_confirmation_and_recovers_after_restart() {
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Test);
        preset.id = "shared".to_string();
        preset.environment = HashMap::from_iter([("TOKEN".to_string(), "top-secret".to_string())]);
        let native_settings = settings::CargoNativeProfilerSettingsContent {
            collector_path: Some("/opt/tools/samply".into()),
            profile_path: Some("target/native-capture.json".into()),
        };
        preset.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(&native_settings)
                .expect("explicit bounded native profiler"),
        );
        let mut project = CargoSettingsContent::default();
        assert!(
            apply_preset_save(
                &mut project,
                &preset,
                CargoPresetSaveScope::Project,
                false,
                true,
            )
            .expect_err("untrusted project save must fail")
            .to_string()
            .contains("Trust")
        );
        assert!(
            apply_preset_save(
                &mut project,
                &preset,
                CargoPresetSaveScope::Project,
                true,
                false,
            )
            .expect_err("unconfirmed project save must fail")
            .to_string()
            .contains("Confirm")
        );
        apply_preset_save(
            &mut project,
            &preset,
            CargoPresetSaveScope::Project,
            true,
            true,
        )
        .expect("confirmed trusted project save should succeed");
        let serialized = serde_json::to_string(&project).expect("settings should serialize");
        assert!(!serialized.contains("top-secret"));
        let restarted: CargoSettingsContent =
            serde_json::from_str(&serialized).expect("saved settings should reload");
        let parsed = parse_settings_content(&restarted);
        assert!(parsed.presets.contains_key("shared"));
        assert_eq!(
            parsed
                .presets
                .get("shared")
                .expect("reloaded preset")
                .native_profiler
                .as_ref()
                .expect("reloaded native declaration")
                .settings_content(),
            native_settings,
        );

        let mut user = CargoSettingsContent::default();
        apply_preset_save(&mut user, &preset, CargoPresetSaveScope::User, true, true)
            .expect("user save should succeed");
        assert_eq!(
            user.presets
                .get("shared")
                .expect("user preset")
                .native_profiler,
            Some(native_settings),
        );
        assert_eq!(
            user.presets["shared"]
                .environment
                .as_ref()
                .and_then(|environment| environment.get("TOKEN"))
                .map(String::as_str),
            Some("top-secret")
        );
    }

    #[gpui::test]
    async fn queued_user_preset_save_reports_revoked_opt_in_without_dismissing(
        cx: &mut TestAppContext,
    ) {
        use fs::Fs as _;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .create_dir(paths::settings_file().parent().expect("settings parent"))
            .await
            .expect("settings directory");
        filesystem
            .insert_file(paths::settings_file(), b"{}".to_vec())
            .await;
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            <dyn fs::Fs>::set_global(filesystem.clone(), cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.id = "queued-experiment".into();
        let (editor, cx) = cx.add_window_view(|window, cx| {
            CargoPresetEditor::new_inner(
                None,
                preset,
                CargoPresetValidationContext::default(),
                window,
                cx,
            )
        });
        let dismissed = Arc::new(AtomicBool::new(false));
        let subscription = cx.update(|_, cx| {
            let dismissed = dismissed.clone();
            cx.subscribe(&editor, move |_, _: &DismissEvent, _| {
                dismissed.store(true, Ordering::SeqCst)
            })
        });
        editor.update(cx, |editor, cx| {
            editor.save_user(cx);
            crate::set_experimental_cargo_opt_ins_for_test(false, cx);
        });
        cx.run_until_parked();
        assert!(!dismissed.load(Ordering::SeqCst));
        assert!(editor.read_with(cx, |editor, _| {
            editor
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("revoked before saving"))
        }));
        let saved = filesystem
            .load(paths::settings_file())
            .await
            .expect("saved settings");
        assert!(!saved.contains("queued-experiment"));
        drop(subscription);
    }

    #[gpui::test]
    async fn queued_user_preset_save_off_then_on_requires_fresh_operation(cx: &mut TestAppContext) {
        use fs::Fs as _;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem
            .create_dir(paths::settings_file().parent().expect("settings parent"))
            .await
            .expect("settings directory");
        filesystem
            .insert_file(
                paths::settings_file(),
                br#"{"feature_flags":{"rust-named-cargo-presets-experimental":"on"}}"#.to_vec(),
            )
            .await;
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            <dyn fs::Fs>::set_global(filesystem.clone(), cx);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.id = "queued-experiment".into();
        let (editor, cx) = cx.add_window_view(|window, cx| {
            CargoPresetEditor::new_inner(
                None,
                preset,
                CargoPresetValidationContext::default(),
                window,
                cx,
            )
        });
        let dismissed = Arc::new(AtomicBool::new(false));
        let subscription = cx.update(|_, cx| {
            let dismissed = dismissed.clone();
            cx.subscribe(&editor, move |_, _: &DismissEvent, _| {
                dismissed.store(true, Ordering::SeqCst)
            })
        });
        editor.update(cx, |editor, cx| {
            editor.save_user(cx);
        });
        cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(false, cx));
        cx.update(|_, cx| crate::set_experimental_cargo_opt_ins_for_test(true, cx));
        assert!(cx.update(|_, cx| cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>()));
        cx.run_until_parked();
        assert!(!dismissed.load(Ordering::SeqCst));
        assert!(editor.read_with(cx, |editor, _| {
            editor
                .last_error
                .as_deref()
                .is_some_and(|error| error.contains("revoked before saving"))
        }));
        let saved = filesystem
            .load(paths::settings_file())
            .await
            .expect("saved settings");
        assert!(!saved.contains("queued-experiment"));
        assert!(cx.update(|_, cx| cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>()));
        editor.update(cx, |editor, cx| editor.save_user(cx));
        cx.run_until_parked();
        assert!(dismissed.load(Ordering::SeqCst));
        assert!(editor.read_with(cx, |editor, _| editor.last_error.is_none()));
        let saved = filesystem
            .load(paths::settings_file())
            .await
            .expect("fresh save completes");
        assert!(saved.contains("queued-experiment"));
        drop(subscription);
    }

    #[gpui::test]
    async fn cargo_preset_editor_keyboard_focus_and_ephemeral_run(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        let (editor, cx) = cx.add_window_view(|window, cx| {
            CargoPresetEditor::new_inner(
                None,
                preset,
                CargoPresetValidationContext::default(),
                window,
                cx,
            )
        });
        editor.update_in(cx, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        cx.run_until_parked();
        cx.update(|window, cx| {
            assert!(editor.read(cx).focus_handle(cx).is_focused(window));
        });

        cx.dispatch_action(RunDraft);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.last_error.clone()),
            Some("Cargo panel is unavailable".to_string())
        );
    }

    #[gpui::test]
    async fn cargo_preset_editor_native_profile_keyboard_action_reaches_current_draft(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(
                &settings::CargoNativeProfilerSettingsContent {
                    collector_path: Some("/opt/tools/samply".into()),
                    profile_path: Some("target/current-capture.json".into()),
                },
            )
            .expect("explicit native declaration"),
        );
        let (editor, cx) = cx.add_window_view(|window, cx| {
            CargoPresetEditor::new_inner(
                None,
                preset,
                CargoPresetValidationContext::default(),
                window,
                cx,
            )
        });
        editor.update_in(cx, |editor, window, cx| {
            window.focus(&editor.focus_handle(cx), cx);
        });
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, cx| editor
                .draft(cx)
                .expect("current editable JSON declaration")
                .profile_action_label()),
            "Native Profile (macOS)"
        );
        cx.dispatch_action(ProfileDraft);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.last_error.clone()),
            Some("Cargo panel is unavailable".to_string())
        );
        cx.update(|_, cx| {
            project::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                false,
                cx,
            );
        });
        cx.dispatch_action(ProfileDraft);
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.last_error.clone()),
            Some("Native profiling requires the rust-native-profiling-experimental opt-in".into())
        );
        assert!(editor.read_with(cx, |editor, cx| editor.draft(cx).is_ok()));
    }

    #[test]
    fn cargo_preset_editor_native_profile_mode_is_declarative_and_redacts_environment() {
        let mut preset = CargoPreset::ephemeral_default(CargoSubcommand::Run);
        preset.native_profiler = Some(
            crate::CargoNativeProfiler::from_settings(
                &settings::CargoNativeProfilerSettingsContent {
                    collector_path: Some("/opt/tools/samply".into()),
                    profile_path: Some("target/capture.json".into()),
                },
            )
            .expect("explicit native declaration"),
        );
        preset
            .environment
            .insert("TOKEN".into(), "sensitive-value".into());
        let content = preset_settings_content(&preset, CargoPresetSaveScope::User);
        let parsed = parse_preset_content(&preset.id, &content)
            .expect("existing editor JSON accepts native mode");
        assert_eq!(parsed.profile_action_label(), "Native Profile (macOS)");
        let preview = redacted_preset_preview(&parsed)
            .expect("native preview")
            .display();
        assert!(preview.contains("Native Profile (macOS)"));
        assert!(preview.contains("Cargo Build then Samply 0.13.1"));
        assert!(preview.contains("TOKEN"));
        assert!(!preview.contains("sensitive-value"));
        assert!(parsed.profiler.is_none());
    }
}
