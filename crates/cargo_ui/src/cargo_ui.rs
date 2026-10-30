mod cargo_actions;
mod cargo_coverage;
mod cargo_panel;
mod cargo_panel_settings;
mod cargo_preset;
mod cargo_preset_editor;
mod cargo_profile;
mod dependency_insight;

pub use cargo_actions::*;
pub use cargo_coverage::*;
pub use cargo_panel::{CargoPanel, CargoTreeProvider, ToggleCargoPanel};
pub use cargo_panel_settings::{CargoPanelSettings, CargoPanelSide};
pub use cargo_preset::*;
pub use cargo_preset_editor::*;
pub use cargo_profile::*;
pub use dependency_insight::*;

use command_palette_hooks::CommandPaletteFilter;
use feature_flags::{
    FeatureFlagAppExt as _, RustCoverageExperimentalFeatureFlag,
    RustNamedCargoPresetsExperimentalFeatureFlag, RustNativeProfilingExperimentalFeatureFlag,
};
use gpui::{App, KeyBinding, UpdateGlobal as _};
use language_tools::language_tool_tree::{
    ActivateSelected, CollapseAll, ExpandAll, Refresh, SelectFirst, SelectFirstChild, SelectLast,
    SelectNext, SelectParent, SelectPrevious, ToggleExpanded,
};
use settings::{CargoPanelSettingsContent, CargoSettingsContent, DockSide, SettingsStore};
use workspace::Workspace;

pub fn init(cx: &mut App) {
    update_experimental_command_filter(cx);
    cx.observe_flag::<RustCoverageExperimentalFeatureFlag, _>(|_, cx| {
        update_experimental_command_filter(cx)
    })
    .detach();
    cx.observe_flag::<RustNativeProfilingExperimentalFeatureFlag, _>(|_, cx| {
        update_experimental_command_filter(cx)
    })
    .detach();
    cx.observe_flag::<RustNamedCargoPresetsExperimentalFeatureFlag, _>(|_, cx| {
        update_experimental_command_filter(cx)
    })
    .detach();
    SettingsStore::update_global(cx, |store, cx| {
        store.update_default_settings(cx, |settings| {
            settings.cargo_panel = Some(CargoPanelSettingsContent {
                button: Some(true),
                default_width: Some(280.),
                dock: Some(DockSide::Right),
                starts_open: Some(false),
            });
            settings.project.cargo = Some(CargoSettingsContent {
                schema_version: Some(CARGO_PRESET_SCHEMA_VERSION),
                presets: Default::default(),
            });
        });
        store.register_setting::<feature_flags::FeatureFlagsSettings>();
        store.register_setting::<CargoPanelSettings>();
        store.register_setting::<CargoPresetSettings>();
    });
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(CargoPanel::toggle_focus);
    })
    .detach();
    bind_keys(cx);
}

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("up", SelectPrevious, Some("CargoPanel")),
        KeyBinding::new("down", SelectNext, Some("CargoPanel")),
        KeyBinding::new("home", SelectFirst, Some("CargoPanel")),
        KeyBinding::new("end", SelectLast, Some("CargoPanel")),
        KeyBinding::new("left", SelectParent, Some("CargoPanel")),
        KeyBinding::new("right", SelectFirstChild, Some("CargoPanel")),
        KeyBinding::new("space", ToggleExpanded, Some("CargoPanel")),
        KeyBinding::new("enter", ActivateSelected, Some("CargoPanel")),
        KeyBinding::new("cmd-shift-e", ExpandAll, Some("CargoPanel")),
        KeyBinding::new("cmd-shift-c", CollapseAll, Some("CargoPanel")),
        KeyBinding::new("cmd-r", Refresh, Some("CargoPanel")),
    ]);
}

#[cfg(all(test, target_os = "macos"))]
#[ctor::ctor(unsafe)]
fn dispatch_native_profile_test_helper() {
    util::macos_task_completion::run_local_task_helper_if_invoked();
}

fn update_experimental_command_filter(cx: &mut App) {
    let coverage = cx.has_flag::<RustCoverageExperimentalFeatureFlag>();
    let native = cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>();
    let presets = cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>();
    CommandPaletteFilter::update_global(cx, |filter, _| {
        for (enabled, action) in [
            (coverage, std::any::TypeId::of::<RunWithCoverageSelected>()),
            (native || presets, std::any::TypeId::of::<ProfileSelected>()),
            (presets, std::any::TypeId::of::<cargo_panel::SelectPreset>()),
            (
                presets,
                std::any::TypeId::of::<cargo_preset_editor::RunDraft>(),
            ),
            (
                presets,
                std::any::TypeId::of::<cargo_preset_editor::DebugDraft>(),
            ),
            (
                presets,
                std::any::TypeId::of::<cargo_preset_editor::SaveDraftForUser>(),
            ),
            (
                presets,
                std::any::TypeId::of::<cargo_preset_editor::SaveDraftForProject>(),
            ),
            (
                presets,
                std::any::TypeId::of::<cargo_preset_editor::ProfileDraft>(),
            ),
        ] {
            if enabled {
                filter.show_action_types([&action]);
            } else {
                filter.hide_action_types([&action]);
            }
        }
    });
}

pub(crate) fn ensure_experimental_cargo_action(
    action: CargoAction,
    preset: Option<&CargoPreset>,
    cx: &App,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        preset.is_none() || cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>(),
        "Named Cargo presets require the rust-named-cargo-presets-experimental opt-in"
    );
    anyhow::ensure!(
        action != CargoAction::RunWithCoverage
            || cx.has_flag::<RustCoverageExperimentalFeatureFlag>(),
        "Rust coverage requires the rust-coverage-experimental opt-in"
    );
    anyhow::ensure!(
        action != CargoAction::Profile
            || !preset.is_some_and(|preset| preset.native_profiler.is_some())
            || cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
        "Native profiling requires the rust-native-profiling-experimental opt-in"
    );
    Ok(())
}

#[cfg(test)]
mod experimental_tests {
    use super::*;

    #[gpui::test]
    async fn cargo_experimental_command_filter_requires_explicit_flags_and_revokes(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            command_palette_hooks::init(cx);
            init(cx);
            cx.update_flags(true, vec![]);
        });
        cx.run_until_parked();
        cx.update(|cx| {
            let filter = CommandPaletteFilter::try_global(cx).expect("existing palette owner");
            assert!(filter.is_hidden(&RunWithCoverageSelected));
            assert!(filter.is_hidden(&ProfileSelected));
            assert!(filter.is_hidden(&cargo_panel::SelectPreset));
            assert!(!filter.is_hidden(&BuildSelected));
            assert!(!filter.is_hidden(&TestSelected));
            assert!(!filter.is_hidden(&TreeSelected));
            crate::set_experimental_cargo_opt_ins_for_test(true, cx);
        });
        cx.run_until_parked();
        cx.update(|cx| {
            let filter = CommandPaletteFilter::try_global(cx).expect("existing palette owner");
            assert!(!filter.is_hidden(&RunWithCoverageSelected));
            assert!(!filter.is_hidden(&ProfileSelected));
            assert!(!filter.is_hidden(&cargo_panel::SelectPreset));
            crate::set_experimental_cargo_opt_ins_for_test(false, cx);
        });
        cx.run_until_parked();
        cx.update(|cx| {
            let filter = CommandPaletteFilter::try_global(cx).expect("existing palette owner");
            assert!(filter.is_hidden(&RunWithCoverageSelected));
            assert!(filter.is_hidden(&ProfileSelected));
            assert!(filter.is_hidden(&cargo_panel::SelectPreset));
            assert!(!filter.is_hidden(&BuildSelected));
        });
    }
}

#[cfg(test)]
pub(crate) fn set_experimental_cargo_opt_ins_for_test(enabled: bool, cx: &mut App) {
    SettingsStore::update_global(cx, |store, _| {
        store.register_setting::<feature_flags::FeatureFlagsSettings>()
    });
    feature_flags::FeatureFlagStore::init(cx);
    SettingsStore::update_global(cx, |store, cx| {
        store.update_user_settings(cx, |content| {
            for name in [
                "rust-coverage-experimental",
                "rust-native-profiling-experimental",
                "rust-named-cargo-presets-experimental",
            ] {
                content
                    .feature_flags
                    .get_or_insert_default()
                    .insert(name.into(), if enabled { "on" } else { "off" }.into());
            }
        });
    });
}
