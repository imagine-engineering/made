use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "agentic")]
use std::time::Duration;

#[cfg(feature = "agentic")]
use client::{Client, zed_urls};
use client::{TelemetrySettings, UserStore};
#[cfg(feature = "agentic")]
use cloud_api_types::Plan;
#[cfg(feature = "agentic")]
use collections::HashMap;
use fs::Fs;
use gpui::{Action, App, Entity, IntoElement, Role};
#[cfg(feature = "agentic")]
use gpui::{Animation, AnimationExt, TaskExt, pulsating_between};
#[cfg(feature = "agentic")]
use project::agent_server_store::AllAgentServersSettings;
use project::project_settings::ProjectSettings;
#[cfg(feature = "agentic")]
use project::{AgentRegistryStore, RegistryAgent};
use settings::{BaseKeymap, Settings, update_settings_file};
#[cfg(feature = "agentic")]
use settings::{CustomAgentServerSettings, SettingsStore};
use theme::{Appearance, SystemAppearance, ThemeRegistry};
use theme_settings::{ThemeAppearanceMode, ThemeName, ThemeSelection, ThemeSettings};
#[cfg(feature = "agentic")]
use ui::AgentSetupButton;
use ui::{
    Divider, StatefulInteractiveElement, SwitchField, TintColor, ToggleButtonGroup,
    ToggleButtonGroupSize, ToggleButtonSimple, ToggleButtonWithIcon, Tooltip, prelude::*,
};
use vim_mode_setting::VimModeSetting;

use crate::{
    ImportCursorSettings, ImportVsCodeSettings, SettingsImportState,
    theme_preview::{ThemePreviewStyle, ThemePreviewTile},
};

fn command_is_available(command: &str) -> bool {
    command_path(command, std::env::var_os("PATH").as_deref()).is_some()
}

fn command_path(command: &str, paths: Option<&OsStr>) -> Option<PathBuf> {
    paths.and_then(|paths| {
        std::env::split_paths(paths).find_map(|directory| command_in_directory(command, &directory))
    })
}

fn command_in_directory(command: &str, directory: &Path) -> Option<PathBuf> {
    command_candidates(command, cfg!(target_os = "windows"))
        .iter()
        .map(|candidate| directory.join(candidate))
        .find(|path| is_usable_command(path))
}

fn command_candidates(command: &str, windows: bool) -> Vec<String> {
    if windows {
        ["exe", "cmd", "bat"]
            .into_iter()
            .map(|extension| format!("{command}.{extension}"))
            .collect()
    } else {
        vec![command.to_string()]
    }
}

fn is_usable_command(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn render_rust_toolchain_section() -> impl IntoElement {
    debug_assert_eq!(product_flavor::TOOLCHAIN_ONBOARDING, "rustup");
    let cargo_status = if command_is_available("cargo") {
        "Cargo detected"
    } else {
        "Cargo not detected"
    };
    let rustup_status = if command_is_available("rustup") {
        "rustup detected"
    } else {
        "rustup not detected"
    };
    let language_servers = product_flavor::DEFAULT_LANGUAGE_SERVERS.join(", ");

    v_flex()
        .gap_2()
        .child(Label::new("Rust toolchain").size(LabelSize::Large))
        .child(
            Label::new(format!(
                "{cargo_status} · {rustup_status} · {language_servers} is built in"
            ))
            .color(Color::Muted),
        )
        .child(
            Label::new(
                "Install or change toolchains explicitly with rustup. Cargo commands are available from the Rust tooling menu.",
            )
            .color(Color::Muted),
        )
        .child(
            Button::new("rustup_docs", "Open rustup setup guide")
                .style(ButtonStyle::Outlined)
                .on_click(|_, _, cx| cx.open_url("https://rustup.rs")),
        )
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum JdkStatus {
    Missing,
    InvalidJavaHome,
    RuntimeOnly,
    VersionUnknown,
    Incompatible(u32),
    Ready(u32),
}

#[derive(Clone)]
struct JvmToolchainStatus {
    jdk: JdkStatus,
    maven_available: bool,
    gradle_available: bool,
}

fn major_java_version(release: &str) -> Option<u32> {
    let value = release
        .lines()
        .find_map(|line| line.strip_prefix("JAVA_VERSION="))?
        .trim_matches('"');
    let major = value.split('.').next()?.parse::<u32>().ok()?;
    if major == 1 {
        value.split('.').nth(1)?.parse().ok()
    } else {
        Some(major)
    }
}

fn classify_jdk(java_available: bool, javac_available: bool, version: Option<u32>) -> JdkStatus {
    if !java_available {
        JdkStatus::Missing
    } else if !javac_available {
        JdkStatus::RuntimeOnly
    } else {
        match version {
            Some(version) if version >= 21 => JdkStatus::Ready(version),
            Some(version) => JdkStatus::Incompatible(version),
            None => JdkStatus::VersionUnknown,
        }
    }
}

fn jdk_status_for_paths(java_home: Option<&Path>, paths: Option<&OsStr>) -> JdkStatus {
    if let Some(java_home) = java_home {
        return command_in_directory("java", &java_home.join("bin"))
            .as_deref()
            .map(jdk_status_for_java_path)
            .unwrap_or(JdkStatus::InvalidJavaHome);
    }
    command_path("java", paths)
        .as_deref()
        .map(jdk_status_for_java_path)
        .unwrap_or(JdkStatus::Missing)
}

fn jdk_status_for_java_path(java_path: &Path) -> JdkStatus {
    let Some(bin_directory) = java_path.parent() else {
        return JdkStatus::VersionUnknown;
    };
    let compiler_available = command_in_directory("javac", bin_directory).is_some();
    let version = bin_directory
        .parent()
        .and_then(|java_home| std::fs::read_to_string(java_home.join("release")).ok())
        .and_then(|release| major_java_version(&release));
    classify_jdk(true, compiler_available, version)
}

fn jdk_status() -> JdkStatus {
    let java_home = std::env::var_os("JAVA_HOME").map(PathBuf::from);
    let paths = std::env::var_os("PATH");
    jdk_status_for_paths(java_home.as_deref(), paths.as_deref())
}

fn jvm_toolchain_status() -> JvmToolchainStatus {
    JvmToolchainStatus {
        jdk: jdk_status(),
        maven_available: command_is_available("mvn"),
        gradle_available: command_is_available("gradle"),
    }
}

fn render_jvm_toolchain_section(status: JvmToolchainStatus) -> impl IntoElement {
    let jdtls_status = match status.jdk {
        JdkStatus::Missing => "Local JDK candidate: missing".to_string(),
        JdkStatus::InvalidJavaHome => {
            "JAVA_HOME does not contain an executable java; configure a JDK 21+".to_string()
        }
        JdkStatus::RuntimeOnly => {
            "Local Java runtime detected, but javac is missing; configure a JDK".to_string()
        }
        JdkStatus::VersionUnknown => {
            "Local JDK candidate: detected, version unverified".to_string()
        }
        JdkStatus::Incompatible(version) => {
            format!("Local JDK candidate: Java {version} detected; JDTLS needs Java 21+")
        }
        JdkStatus::Ready(version) => format!("Local JDK candidate: Java {version} detected"),
    };
    let maven_status = if status.maven_available {
        "Maven detected"
    } else {
        "Maven not detected"
    };
    let gradle_status = if status.gradle_available {
        "Gradle detected"
    } else {
        "Gradle not detected"
    };
    let accessibility_summary = format!(
        "Groovy JVM setup. {jdtls_status}. {maven_status}. {gradle_status}. Local tool check only; an overridden JDTLS path or remote host is not checked. Setup does not install tools or run build scripts."
    );

    v_flex()
        .id("groovy-jvm-toolchain-status")
        .role(Role::Group)
        .aria_label(accessibility_summary)
        .gap_2()
        .child(Label::new("Groovy JVM setup").size(LabelSize::Large))
        .child(Label::new(jdtls_status).color(Color::Muted))
        .child(Label::new(format!("{maven_status} · {gradle_status}")).color(Color::Muted))
        .child(
            Label::new(
                "Use a JDK 21+ with java and javac in JAVA_HOME or the same PATH directory to verify the local JDTLS candidate. This does not check an overridden JDTLS path or a remote host. Project wrappers may supply Maven or Gradle; the build JDK and target JDK can differ. Setup does not install tools or run build scripts.",
            )
            .color(Color::Muted),
        )
        .child(
            h_flex()
                .flex_wrap()
                .gap_2()
                .child(
                    div().debug_selector(|| "groovy-jdk-guide".to_string()).child(
                        Button::new("groovy-jdk-guide-button", "Open Temurin JDK guide")
                            .style(ButtonStyle::Outlined)
                            .on_click(|_, _, cx| cx.open_url("https://adoptium.net/installation")),
                    ),
                )
                .when(!status.maven_available, |this| {
                    this.child(
                        div().debug_selector(|| "groovy-maven-guide".to_string()).child(
                            Button::new("groovy-maven-guide-button", "Open Maven guide")
                                .style(ButtonStyle::Outlined)
                                .on_click(|_, _, cx| cx.open_url("https://maven.apache.org/install")),
                        ),
                    )
                })
                .when(!status.gradle_available, |this| {
                    this.child(
                        div().debug_selector(|| "groovy-gradle-guide".to_string()).child(
                            Button::new("groovy-gradle-guide-button", "Open Gradle guide")
                                .style(ButtonStyle::Outlined)
                                .on_click(|_, _, cx| {
                                    cx.open_url("https://docs.gradle.org/current/userguide/installation.html")
                                }),
                        ),
                    )
                })
                .child(
                    Button::new("groovy-recheck-tools", "Check local tools again")
                        .style(ButtonStyle::Outlined)
                        .on_click(|_, _, cx| cx.refresh_windows()),
                ),
        )
}

const LIGHT_THEMES: [&str; 3] = ["One Light", "Ayu Light", "Gruvbox Light"];
const DARK_THEMES: [&str; 3] = ["One Dark", "Ayu Dark", "Gruvbox Dark"];
const BASE_KEYMAP_OPTION_LABELS: [&str; 8] = [
    "Zed",
    "VS Code",
    "JetBrains",
    "Sublime Text",
    "Atom",
    "Emacs",
    "Cursor",
    "TextMate",
];
const FAMILY_NAMES: [SharedString; 3] = [
    SharedString::new_static("One"),
    SharedString::new_static("Ayu"),
    SharedString::new_static("Gruvbox"),
];

fn get_theme_family_themes(theme_name: &str) -> Option<(&'static str, &'static str)> {
    for i in 0..LIGHT_THEMES.len() {
        if LIGHT_THEMES[i] == theme_name || DARK_THEMES[i] == theme_name {
            return Some((LIGHT_THEMES[i], DARK_THEMES[i]));
        }
    }
    None
}

fn render_theme_section(tab_index: &mut isize, cx: &mut App) -> impl IntoElement {
    let theme_selection = ThemeSettings::get_global(cx).theme.clone();
    let system_appearance = theme::SystemAppearance::global(cx);

    let theme_mode = theme_selection
        .mode()
        .unwrap_or_else(|| match *system_appearance {
            Appearance::Light => ThemeAppearanceMode::Light,
            Appearance::Dark => ThemeAppearanceMode::Dark,
        });

    return v_flex()
        .gap_2()
        .child(
            h_flex().justify_between().child(Label::new("Theme")).child(
                ToggleButtonGroup::single_row(
                    "theme-selector-onboarding-dark-light",
                    [
                        ThemeAppearanceMode::Light,
                        ThemeAppearanceMode::Dark,
                        ThemeAppearanceMode::System,
                    ]
                    .map(|mode| {
                        const MODE_NAMES: [SharedString; 3] = [
                            SharedString::new_static("Light"),
                            SharedString::new_static("Dark"),
                            SharedString::new_static("System"),
                        ];
                        ToggleButtonSimple::new(
                            MODE_NAMES[mode as usize].clone(),
                            move |_, _, cx| {
                                write_mode_change(mode, cx);

                                telemetry::event!(
                                    "Welcome Theme mode Changed",
                                    from = theme_mode,
                                    to = mode
                                );
                            },
                        )
                    }),
                )
                .size(ToggleButtonGroupSize::Medium)
                .tab_index(tab_index)
                .selected_index(theme_mode as usize)
                .style(ui::ToggleButtonGroupStyle::Outlined)
                .width(rems_from_px(3.0_f32 * 64.0_f32)),
            ),
        )
        .child(
            h_flex()
                .gap_2()
                .justify_between()
                .children(render_theme_previews(tab_index, &theme_selection, cx)),
        );

    fn render_theme_previews(
        tab_index: &mut isize,
        theme_selection: &ThemeSelection,
        cx: &mut App,
    ) -> [impl IntoElement; 3] {
        let system_appearance = SystemAppearance::global(cx);
        let theme_registry = ThemeRegistry::global(cx);

        let theme_seed = 0xBEEF as f32;
        let theme_mode = theme_selection
            .mode()
            .unwrap_or_else(|| match *system_appearance {
                Appearance::Light => ThemeAppearanceMode::Light,
                Appearance::Dark => ThemeAppearanceMode::Dark,
            });
        let appearance = match theme_mode {
            ThemeAppearanceMode::Light => Appearance::Light,
            ThemeAppearanceMode::Dark => Appearance::Dark,
            ThemeAppearanceMode::System => *system_appearance,
        };
        let current_theme_name: SharedString = theme_selection.name(appearance).0.into();

        let theme_names = match appearance {
            Appearance::Light => LIGHT_THEMES,
            Appearance::Dark => DARK_THEMES,
        };

        let themes = theme_names.map(|theme| theme_registry.get(theme).unwrap());

        [0, 1, 2].map(|index| {
            let theme = &themes[index];
            let is_selected = theme.name == current_theme_name;
            let name = theme.name.clone();
            let colors = cx.theme().colors();

            v_flex()
                .w_full()
                .items_center()
                .gap_1()
                .child(
                    h_flex()
                        .id(name)
                        .relative()
                        .w_full()
                        .border_2()
                        .border_color(colors.border_transparent)
                        .rounded(ThemePreviewTile::ROOT_RADIUS)
                        .map(|this| {
                            if is_selected {
                                this.border_color(colors.border_selected)
                            } else {
                                this.opacity(0.8).hover(|s| s.border_color(colors.border))
                            }
                        })
                        .tab_index({
                            *tab_index += 1;
                            *tab_index - 1
                        })
                        .focus(|mut style| {
                            style.border_color = Some(colors.border_focused);
                            style
                        })
                        .on_click({
                            let theme_name = theme.name.clone();
                            let current_theme_name = current_theme_name.clone();

                            move |_, _, cx| {
                                write_theme_change(theme_name.clone(), theme_mode, cx);
                                telemetry::event!(
                                    "Welcome Theme Changed",
                                    from = current_theme_name,
                                    to = theme_name
                                );
                            }
                        })
                        .map(|this| {
                            if theme_mode == ThemeAppearanceMode::System {
                                let (light, dark) = (
                                    theme_registry.get(LIGHT_THEMES[index]).unwrap(),
                                    theme_registry.get(DARK_THEMES[index]).unwrap(),
                                );
                                this.child(
                                    ThemePreviewTile::new(light, theme_seed)
                                        .style(ThemePreviewStyle::SideBySide(dark)),
                                )
                            } else {
                                this.child(
                                    ThemePreviewTile::new(theme.clone(), theme_seed)
                                        .style(ThemePreviewStyle::Bordered),
                                )
                            }
                        }),
                )
                .child(
                    Label::new(FAMILY_NAMES[index].clone())
                        .color(Color::Muted)
                        .size(LabelSize::Small),
                )
        })
    }

    fn write_mode_change(mode: ThemeAppearanceMode, cx: &mut App) {
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, move |settings, _cx| {
            theme_settings::set_mode(settings, mode);
        });
    }

    fn write_theme_change(
        theme: impl Into<Arc<str>>,
        theme_mode: ThemeAppearanceMode,
        cx: &mut App,
    ) {
        let fs = <dyn Fs>::global(cx);
        let theme = theme.into();
        update_settings_file(fs, cx, move |settings, cx| match theme_mode {
            ThemeAppearanceMode::System => {
                let (light_theme, dark_theme) =
                    get_theme_family_themes(&theme).unwrap_or((theme.as_ref(), theme.as_ref()));

                settings.theme.theme = Some(settings::ThemeSelection::Dynamic {
                    mode: ThemeAppearanceMode::System,
                    light: ThemeName(light_theme.into()),
                    dark: ThemeName(dark_theme.into()),
                });
            }
            ThemeAppearanceMode::Light => theme_settings::set_theme(
                settings,
                theme,
                Appearance::Light,
                *SystemAppearance::global(cx),
            ),
            ThemeAppearanceMode::Dark => theme_settings::set_theme(
                settings,
                theme,
                Appearance::Dark,
                *SystemAppearance::global(cx),
            ),
        });
    }
}

fn render_telemetry_section(tab_index: &mut isize, cx: &App) -> impl IntoElement {
    let fs = <dyn Fs>::global(cx);

    v_flex()
        .gap_4()
        .child(
            SwitchField::new(
                "onboarding-telemetry-metrics",
                None::<&str>,
                Some(
                    format!(
                        "Help improve {} by sending anonymous usage data",
                        product_flavor::DISPLAY_NAME
                    )
                    .into(),
                ),
                if TelemetrySettings::get_global(cx).metrics {
                    ui::ToggleState::Selected
                } else {
                    ui::ToggleState::Unselected
                },
                {
                    let fs = fs.clone();
                    move |selection, _, cx| {
                        let enabled = match selection {
                            ToggleState::Selected => true,
                            ToggleState::Unselected => false,
                            ToggleState::Indeterminate => {
                                return;
                            }
                        };

                        update_settings_file(fs.clone(), cx, move |setting, _| {
                            setting.telemetry.get_or_insert_default().metrics = Some(enabled);
                        });

                        // This telemetry event shouldn't fire when it's off. If it does we'll be alerted
                        // and can fix it in a timely manner to respect a user's choice.
                        telemetry::event!(
                            "Welcome Page Telemetry Metrics Toggled",
                            options = if enabled { "on" } else { "off" }
                        );
                    }
                },
            )
            .tab_index({
                *tab_index += 1;
                *tab_index
            }),
        )
        .child(
            SwitchField::new(
                "onboarding-telemetry-crash-reports",
                None::<&str>,
                Some(
                    format!(
                        "Help fix {} by sending crash reports so we can fix critical issues fast",
                        product_flavor::DISPLAY_NAME
                    )
                    .into(),
                ),
                if TelemetrySettings::get_global(cx).diagnostics {
                    ui::ToggleState::Selected
                } else {
                    ui::ToggleState::Unselected
                },
                {
                    let fs = fs.clone();
                    move |selection, _, cx| {
                        let enabled = match selection {
                            ToggleState::Selected => true,
                            ToggleState::Unselected => false,
                            ToggleState::Indeterminate => {
                                return;
                            }
                        };

                        update_settings_file(fs.clone(), cx, move |setting, _| {
                            setting.telemetry.get_or_insert_default().diagnostics = Some(enabled);
                        });

                        // This telemetry event shouldn't fire when it's off. If it does we'll be alerted
                        // and can fix it in a timely manner to respect a user's choice.
                        telemetry::event!(
                            "Welcome Page Telemetry Diagnostics Toggled",
                            options = if enabled { "on" } else { "off" }
                        );
                    }
                },
            )
            .tab_index({
                *tab_index += 1;
                *tab_index
            }),
        )
}

fn render_base_keymap_section(tab_index: &mut isize, cx: &mut App) -> impl IntoElement {
    let [
        zed,
        vscode,
        jetbrains,
        sublime_text,
        atom,
        emacs,
        cursor,
        textmate,
    ] = BASE_KEYMAP_OPTION_LABELS;
    let base_keymap = match BaseKeymap::get_global(cx) {
        BaseKeymap::Zed => Some(0),
        BaseKeymap::VSCode => Some(1),
        BaseKeymap::JetBrains => Some(2),
        BaseKeymap::SublimeText => Some(3),
        BaseKeymap::Atom => Some(4),
        BaseKeymap::Emacs => Some(5),
        BaseKeymap::Cursor => Some(6),
        BaseKeymap::TextMate => Some(7),
        BaseKeymap::None => None,
    };

    return v_flex().gap_2().child(Label::new("Base Keymap")).child(
        ToggleButtonGroup::two_rows(
            "base_keymap_selection",
            [
                ToggleButtonWithIcon::new(zed, IconName::AiZed, |_, _, cx| {
                    write_keymap_base(BaseKeymap::Zed, cx);
                }),
                ToggleButtonWithIcon::new(vscode, IconName::EditorVsCode, |_, _, cx| {
                    write_keymap_base(BaseKeymap::VSCode, cx);
                }),
                ToggleButtonWithIcon::new(jetbrains, IconName::EditorJetBrains, |_, _, cx| {
                    write_keymap_base(BaseKeymap::JetBrains, cx);
                }),
                ToggleButtonWithIcon::new(sublime_text, IconName::EditorSublime, |_, _, cx| {
                    write_keymap_base(BaseKeymap::SublimeText, cx);
                }),
            ],
            [
                ToggleButtonWithIcon::new(atom, IconName::EditorAtom, |_, _, cx| {
                    write_keymap_base(BaseKeymap::Atom, cx);
                }),
                ToggleButtonWithIcon::new(emacs, IconName::EditorEmacs, |_, _, cx| {
                    write_keymap_base(BaseKeymap::Emacs, cx);
                }),
                ToggleButtonWithIcon::new(cursor, IconName::EditorCursor, |_, _, cx| {
                    write_keymap_base(BaseKeymap::Cursor, cx);
                }),
                ToggleButtonWithIcon::new(textmate, IconName::Keyboard, |_, _, cx| {
                    write_keymap_base(BaseKeymap::TextMate, cx);
                }),
            ],
        )
        .when_some(base_keymap, |this, base_keymap| {
            this.selected_index(base_keymap)
        })
        .full_width()
        .tab_index(tab_index)
        .size(ui::ToggleButtonGroupSize::Medium)
        .style(ui::ToggleButtonGroupStyle::Outlined),
    );

    fn write_keymap_base(keymap_base: BaseKeymap, cx: &App) {
        let fs = <dyn Fs>::global(cx);

        update_settings_file(fs, cx, move |setting, _| {
            setting.base_keymap = Some(keymap_base.into());
        });

        telemetry::event!("Welcome Keymap Changed", keymap = keymap_base);
    }
}

fn render_vim_mode_switch(tab_index: &mut isize, cx: &mut App) -> impl IntoElement {
    let toggle_state = if VimModeSetting::get_global(cx).0 {
        ui::ToggleState::Selected
    } else {
        ui::ToggleState::Unselected
    };
    SwitchField::new(
        "onboarding-vim-mode",
        Some("Vim Mode"),
        Some("Coming from Neovim? Use our first-class implementation of Vim Mode".into()),
        toggle_state,
        {
            let fs = <dyn Fs>::global(cx);
            move |&selection, _, cx| {
                let vim_mode = match selection {
                    ToggleState::Selected => true,
                    ToggleState::Unselected => false,
                    ToggleState::Indeterminate => {
                        return;
                    }
                };
                update_settings_file(fs.clone(), cx, move |setting, _| {
                    setting.vim_mode = Some(vim_mode);
                });

                telemetry::event!(
                    "Welcome Vim Mode Toggled",
                    options = if vim_mode { "on" } else { "off" },
                );
            }
        },
    )
    .tab_index({
        *tab_index += 1;
        *tab_index - 1
    })
}

fn render_worktree_auto_trust_switch(tab_index: &mut isize, cx: &mut App) -> impl IntoElement {
    let toggle_state = if ProjectSettings::get_global(cx).session.trust_all_worktrees {
        ui::ToggleState::Selected
    } else {
        ui::ToggleState::Unselected
    };

    let tooltip_description = format!(
        "{} can only allow services like language servers, project settings, and MCP servers to run after you mark a new project as trusted.",
        product_flavor::DISPLAY_NAME
    );

    SwitchField::new(
        "onboarding-auto-trust-worktrees",
        Some("Trust All Projects By Default"),
        Some(
            format!(
                "Automatically mark all new projects as trusted to unlock all {} features",
                product_flavor::DISPLAY_NAME
            )
            .into(),
        ),
        toggle_state,
        {
            let fs = <dyn Fs>::global(cx);
            move |&selection, _, cx| {
                let trust = match selection {
                    ToggleState::Selected => true,
                    ToggleState::Unselected => false,
                    ToggleState::Indeterminate => {
                        return;
                    }
                };
                update_settings_file(fs.clone(), cx, move |setting, _| {
                    setting.session.get_or_insert_default().trust_all_worktrees = Some(trust);
                });

                telemetry::event!(
                    "Welcome Page Worktree Auto Trust Toggled",
                    options = if trust { "on" } else { "off" }
                );
            }
        },
    )
    .tab_index({
        *tab_index += 1;
        *tab_index - 1
    })
    .tooltip(Tooltip::text(tooltip_description))
}

fn render_setting_import_button(
    tab_index: isize,
    label: SharedString,
    action: &dyn Action,
    imported: bool,
) -> impl IntoElement + 'static {
    let action = action.boxed_clone();

    Button::new(label.clone(), label.clone())
        .style(ButtonStyle::OutlinedGhost)
        .size(ButtonSize::Medium)
        .label_size(LabelSize::Small)
        .selected_style(ButtonStyle::Tinted(TintColor::Accent))
        .toggle_state(imported)
        .tab_index(tab_index)
        .when(imported, |this| {
            this.end_icon(Icon::new(IconName::Check).size(IconSize::Small))
                .color(Color::Success)
        })
        .on_click(move |_, window, cx| {
            telemetry::event!("Welcome Import Settings", import_source = label,);
            window.dispatch_action(action.boxed_clone(), cx);
        })
}

fn render_import_settings_section(tab_index: &mut isize, cx: &mut App) -> impl IntoElement {
    let import_state = SettingsImportState::global(cx);
    let imports: [(SharedString, &dyn Action, bool); 2] = [
        (
            "VS Code".into(),
            &ImportVsCodeSettings { skip_prompt: false },
            import_state.vscode,
        ),
        (
            "Cursor".into(),
            &ImportCursorSettings { skip_prompt: false },
            import_state.cursor,
        ),
    ];

    let [vscode, cursor] = imports.map(|(label, action, imported)| {
        *tab_index += 1;
        render_setting_import_button(*tab_index - 1, label, action, imported)
    });

    h_flex()
        .gap_2()
        .flex_wrap()
        .justify_between()
        .child(
            v_flex()
                .gap_0p5()
                .max_w_5_6()
                .child(Label::new("Import Settings"))
                .child(
                    Label::new("Automatically pull your settings from other editors")
                        .color(Color::Muted),
                ),
        )
        .child(h_flex().gap_1().child(vscode).child(cursor))
}

#[cfg(feature = "agentic")]
pub(crate) const FEATURED_AGENT_IDS: &[&str] =
    &["claude-acp", "codex-acp", "github-copilot-cli", "cursor"];

#[cfg(feature = "agentic")]
fn render_registry_agent_button(
    agent: &RegistryAgent,
    installed: bool,
    cx: &mut App,
) -> impl IntoElement {
    let agent_id = agent.id().to_string();
    let element_id = format!("{}-onboarding", agent_id);

    let icon = match agent.icon_path() {
        Some(icon_path) => Icon::from_external_svg(icon_path.clone()),
        None => Icon::new(IconName::Sparkle),
    }
    .size(IconSize::XSmall)
    .color(Color::Muted);

    let fs = <dyn Fs>::global(cx);

    let state_element = if installed {
        Icon::new(IconName::Check)
            .size(IconSize::Small)
            .color(Color::Success)
            .into_any_element()
    } else {
        Label::new("Install")
            .size(LabelSize::XSmall)
            .color(Color::Muted)
            .into_any_element()
    };

    AgentSetupButton::new(element_id)
        .icon(icon)
        .name(agent.name().clone())
        .state(state_element)
        .disabled(installed)
        .on_click(move |_, window, cx| {
            telemetry::event!("Welcome Agent Install Clicked", agent = agent_id.as_str());
            update_settings_file(fs.clone(), cx, {
                let agent_id = agent_id.clone();
                move |settings, _| {
                    let agent_servers = settings.agent_servers.get_or_insert_default();
                    agent_servers.entry(agent_id).or_insert_with(|| {
                        CustomAgentServerSettings::Registry {
                            env: Default::default(),
                            default_mode: None,
                            default_config_options: HashMap::default(),
                            favorite_config_option_values: HashMap::default(),
                        }
                    });
                }
            });
            window.dispatch_action(
                Box::new(zed_actions::agent::SelectAgent {
                    agent: agent_id.clone(),
                }),
                cx,
            );
        })
}

#[cfg(feature = "agentic")]
fn render_zed_agent_button(user_store: &Entity<UserStore>, cx: &mut App) -> impl IntoElement {
    let client = Client::global(cx);
    let status = *client.status().borrow();

    let plan = user_store.read(cx).plan();
    let is_free = matches!(plan, Some(Plan::ZedFree) | None);
    let is_pro = matches!(plan, Some(Plan::ZedPro));
    let is_trial = matches!(plan, Some(Plan::ZedProTrial));

    let is_signed_out = status.is_signed_out()
        || matches!(
            status,
            client::Status::AuthenticationError | client::Status::ConnectionError
        );
    let is_signing_in = status.is_signing_in();
    let is_signed_in = !is_signed_out;

    let state_element = if is_signed_out {
        Label::new("Sign In")
            .size(LabelSize::XSmall)
            .color(Color::Muted)
            .into_any_element()
    } else if is_signing_in {
        Label::new("Signing In…")
            .size(LabelSize::XSmall)
            .color(Color::Muted)
            .with_animation(
                "signing-in",
                Animation::new(Duration::from_secs(2))
                    .repeat()
                    .with_easing(pulsating_between(0.4, 0.8)),
                |label, delta| label.alpha(delta),
            )
            .into_any_element()
    } else if is_signed_in && is_free {
        Label::new("Start Free Trial")
            .size(LabelSize::XSmall)
            .color(Color::Muted)
            .into_any_element()
    } else {
        Icon::new(IconName::Check)
            .size(IconSize::Small)
            .color(Color::Success)
            .into_any_element()
    };

    AgentSetupButton::new("zed-agent-onboarding")
        .icon(
            Icon::new(IconName::ZedAgent)
                .size(IconSize::XSmall)
                .color(Color::Muted),
        )
        .name("Zed Agent")
        .state(state_element)
        .disabled(is_trial || is_pro)
        .map(|this| {
            if is_signed_in && is_free {
                this.on_click(move |_, _window, cx| {
                    telemetry::event!("Start Trial Clicked", state = "post-sign-in");
                    cx.open_url(&zed_urls::start_trial_url(cx))
                })
            } else {
                this.on_click(move |_, _, cx| {
                    telemetry::event!("Welcome Zed Agent Sign In Clicked");
                    let client = Client::global(cx);
                    cx.spawn(async move |cx| client.sign_in_with_optional_connect(true, cx).await)
                        .detach_and_log_err(cx);
                })
            }
        })
}

#[cfg(feature = "agentic")]
fn render_ai_section(user_store: &Entity<UserStore>, cx: &mut App) -> impl IntoElement {
    let registry_agents = AgentRegistryStore::try_global(cx)
        .map(|store| store.read(cx).agents().to_vec())
        .unwrap_or_default();

    let installed_agents = cx
        .global::<SettingsStore>()
        .get::<AllAgentServersSettings>(None)
        .clone();

    let column_count = 1 + FEATURED_AGENT_IDS.len() as u16;

    let grid = FEATURED_AGENT_IDS.iter().fold(
        div()
            .w_full()
            .mt_1p5()
            .grid()
            .grid_cols(column_count)
            .gap_2()
            .child(render_zed_agent_button(user_store, cx)),
        |grid, agent_id| {
            let Some(agent) = registry_agents
                .iter()
                .find(|a| a.id().as_ref() == *agent_id)
            else {
                return grid;
            };
            let is_installed = installed_agents.contains_key(*agent_id);
            grid.child(render_registry_agent_button(agent, is_installed, cx))
        },
    );

    v_flex()
        .gap_0p5()
        .child(Label::new("Agent Setup"))
        .child(
            Label::new("Install your favorite agents and start your first thread.")
                .color(Color::Muted),
        )
        .child(grid)
}

pub(crate) fn render_basics_page(user_store: &Entity<UserStore>, cx: &mut App) -> impl IntoElement {
    let mut tab_index = 0;

    let page = v_flex()
        .id("basics-page")
        .gap_6()
        .children(crate::workspace_choice::render_workspace_choice(
            &mut tab_index,
            cx,
        ))
        .when(product_flavor::TOOLCHAIN_ONBOARDING == "rustup", |this| {
            this.child(render_rust_toolchain_section())
        })
        .when(product_flavor::TOOLCHAIN_ONBOARDING == "jdk", |this| {
            this.child(render_jvm_toolchain_section(jvm_toolchain_status()))
        })
        .child(render_theme_section(&mut tab_index, cx))
        .child(render_base_keymap_section(&mut tab_index, cx));
    #[cfg(feature = "agentic")]
    let page = page.child(render_ai_section(user_store, cx));
    #[cfg(not(feature = "agentic"))]
    let _ = user_store;
    page.child(render_import_settings_section(&mut tab_index, cx))
        .child(render_vim_mode_switch(&mut tab_index, cx))
        .child(render_worktree_auto_trust_switch(&mut tab_index, cx))
        .child(Divider::horizontal().color(ui::DividerColor::BorderVariant))
        .child(render_telemetry_section(&mut tab_index, cx))
}

#[cfg(test)]
mod tests {
    use super::{
        BASE_KEYMAP_OPTION_LABELS, JdkStatus, JvmToolchainStatus, classify_jdk, command_candidates,
        command_path, is_usable_command, jdk_status_for_paths, major_java_version,
        render_jvm_toolchain_section,
    };
    use gpui::{Context, Modifiers, Render, TestAppContext, Window};

    struct JvmToolchainTestView {
        status: JvmToolchainStatus,
    }

    impl Render for JvmToolchainTestView {
        fn render(
            &mut self,
            _window: &mut Window,
            _cx: &mut Context<Self>,
        ) -> impl gpui::IntoElement {
            render_jvm_toolchain_section(self.status.clone())
        }
    }

    #[gpui::test]
    fn jvm_setup_guides_only_open_after_a_user_action(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let (_, cx) = cx.add_window_view(|_, _| JvmToolchainTestView {
            status: JvmToolchainStatus {
                jdk: JdkStatus::Missing,
                maven_available: false,
                gradle_available: false,
            },
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        assert_eq!(cx.opened_url(), None);
        assert!(cx.debug_bounds("groovy-maven-guide").is_some());
        assert!(cx.debug_bounds("groovy-gradle-guide").is_some());
        let jdk_guide = cx.debug_bounds("groovy-jdk-guide").expect("JDK guide");
        cx.simulate_click(jdk_guide.center(), Modifiers::default());
        assert_eq!(
            cx.opened_url().as_deref(),
            Some("https://adoptium.net/installation")
        );
        let maven_guide = cx.debug_bounds("groovy-maven-guide").expect("Maven guide");
        cx.simulate_click(maven_guide.center(), Modifiers::default());
        assert_eq!(
            cx.opened_url().as_deref(),
            Some("https://maven.apache.org/install")
        );
        let gradle_guide = cx
            .debug_bounds("groovy-gradle-guide")
            .expect("Gradle guide");
        cx.simulate_click(gradle_guide.center(), Modifiers::default());
        assert_eq!(
            cx.opened_url().as_deref(),
            Some("https://docs.gradle.org/current/userguide/installation.html")
        );
    }

    #[gpui::test]
    fn jvm_setup_keeps_build_tool_guides_optional(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let (_, cx) = cx.add_window_view(|_, _| JvmToolchainTestView {
            status: JvmToolchainStatus {
                jdk: JdkStatus::Ready(21),
                maven_available: true,
                gradle_available: true,
            },
        });
        cx.update(|window, cx| window.draw(cx).clear(cx));
        assert!(cx.debug_bounds("groovy-jdk-guide").is_some());
        assert!(cx.debug_bounds("groovy-maven-guide").is_none());
        assert!(cx.debug_bounds("groovy-gradle-guide").is_none());
    }

    #[test]
    fn base_keymap_options_use_upstream_zed_name() {
        assert!(BASE_KEYMAP_OPTION_LABELS.contains(&"Zed"));
        assert!(!BASE_KEYMAP_OPTION_LABELS.contains(&"Sim"));
    }

    #[test]
    fn jdk_release_version_distinguishes_legacy_and_current_runtimes() {
        assert_eq!(major_java_version("JAVA_VERSION=\"1.8.0_402\""), Some(8));
        assert_eq!(major_java_version("JAVA_VERSION=\"17.0.19\""), Some(17));
        assert_eq!(major_java_version("JAVA_VERSION=\"21.0.6\""), Some(21));
        assert_eq!(major_java_version("NOT_A_RELEASE=\"21\""), None);
    }

    #[test]
    fn toolchain_command_candidates_include_windows_build_launchers() {
        assert_eq!(command_candidates("mvn", false), ["mvn"]);
        assert_eq!(
            command_candidates("mvn", true),
            ["mvn.exe", "mvn.cmd", "mvn.bat"]
        );
        assert_eq!(
            command_candidates("gradle", true),
            ["gradle.exe", "gradle.cmd", "gradle.bat"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn toolchain_probe_requires_an_executable_file() {
        let executable = std::env::current_exe().expect("current test executable");
        assert!(is_usable_command(&executable));
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/basics_page.rs");
        assert!(!is_usable_command(&source));
    }

    #[test]
    fn jdtls_setup_requires_a_verified_java_21_runtime() {
        assert_eq!(classify_jdk(false, false, None), JdkStatus::Missing);
        assert_eq!(classify_jdk(true, false, Some(21)), JdkStatus::RuntimeOnly);
        assert_eq!(classify_jdk(true, true, None), JdkStatus::VersionUnknown);
        assert_eq!(
            classify_jdk(true, true, Some(17)),
            JdkStatus::Incompatible(17)
        );
        assert_eq!(classify_jdk(true, true, Some(21)), JdkStatus::Ready(21));
    }

    #[cfg(unix)]
    #[test]
    fn path_jdk_probe_keeps_java_and_javac_in_one_installation() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().expect("temporary toolchain directory");
        let runtime_bin = directory.path().join("runtime/bin");
        let compiler_bin = directory.path().join("compiler/bin");
        let jdk_bin = directory.path().join("jdk/bin");
        for bin in [&runtime_bin, &compiler_bin, &jdk_bin] {
            std::fs::create_dir_all(bin).expect("toolchain bin directory");
        }
        for executable in [
            runtime_bin.join("java"),
            compiler_bin.join("javac"),
            jdk_bin.join("java"),
            jdk_bin.join("javac"),
        ] {
            std::fs::write(&executable, "").expect("toolchain executable");
            std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755))
                .expect("executable permission");
        }
        let runtime_home = directory.path().join("runtime");
        let jdk_home = directory.path().join("jdk");
        std::fs::write(jdk_home.join("release"), "JAVA_VERSION=\"21.0.6\"\n")
            .expect("JDK release metadata");

        let mixed_path =
            std::env::join_paths([&runtime_bin, &compiler_bin]).expect("mixed toolchain PATH");
        std::fs::write(runtime_bin.join("cargo"), "").expect("nonexecutable Cargo candidate");
        let cargo = compiler_bin.join("cargo");
        std::fs::write(&cargo, "").expect("executable Cargo candidate");
        std::fs::set_permissions(&cargo, std::fs::Permissions::from_mode(0o755))
            .expect("Cargo executable permission");
        assert_eq!(command_path("cargo", Some(&mixed_path)), Some(cargo));
        assert_eq!(
            jdk_status_for_paths(None, Some(&mixed_path)),
            JdkStatus::RuntimeOnly
        );
        let jdk_path = std::env::join_paths([&jdk_bin]).expect("JDK PATH");
        assert_eq!(
            jdk_status_for_paths(None, Some(&jdk_path)),
            JdkStatus::Ready(21)
        );
        assert_eq!(
            jdk_status_for_paths(Some(&runtime_home), Some(&jdk_path)),
            JdkStatus::RuntimeOnly,
            "JAVA_HOME takes precedence over a different PATH JDK"
        );
        let missing_home = directory.path().join("missing");
        assert_eq!(
            jdk_status_for_paths(Some(&missing_home), Some(&jdk_path)),
            JdkStatus::InvalidJavaHome,
            "a broken JAVA_HOME must not inherit an unrelated PATH JDK"
        );
        std::fs::write(jdk_home.join("release"), "JAVA_VERSION=\"17.0.19\"\n")
            .expect("older JDK release metadata");
        assert_eq!(
            jdk_status_for_paths(None, Some(&jdk_path)),
            JdkStatus::Incompatible(17)
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_path_jdk_probe_keeps_java_and_javac_in_one_installation() {
        let directory = tempfile::tempdir().expect("temporary toolchain directory");
        let runtime_bin = directory.path().join("runtime/bin");
        let compiler_bin = directory.path().join("compiler/bin");
        let jdk_bin = directory.path().join("jdk/bin");
        for bin in [&runtime_bin, &compiler_bin, &jdk_bin] {
            std::fs::create_dir_all(bin).expect("toolchain bin directory");
        }
        for executable in [
            runtime_bin.join("java.exe"),
            compiler_bin.join("javac.exe"),
            jdk_bin.join("java.exe"),
            jdk_bin.join("javac.exe"),
        ] {
            std::fs::write(executable, "").expect("toolchain executable candidate");
        }
        let jdk_home = directory.path().join("jdk");
        std::fs::write(jdk_home.join("release"), "JAVA_VERSION=\"21.0.6\"\n")
            .expect("JDK release metadata");

        let mixed_path =
            std::env::join_paths([&runtime_bin, &compiler_bin]).expect("mixed toolchain PATH");
        assert_eq!(
            jdk_status_for_paths(None, Some(&mixed_path)),
            JdkStatus::RuntimeOnly
        );
        let jdk_path = std::env::join_paths([&jdk_bin]).expect("JDK PATH");
        assert_eq!(
            jdk_status_for_paths(None, Some(&jdk_path)),
            JdkStatus::Ready(21)
        );
        assert_eq!(
            jdk_status_for_paths(Some(&directory.path().join("runtime")), Some(&jdk_path)),
            JdkStatus::RuntimeOnly
        );
        assert_eq!(
            jdk_status_for_paths(Some(&directory.path().join("missing")), Some(&jdk_path)),
            JdkStatus::InvalidJavaHome
        );

        let maven = jdk_bin.join("mvn.cmd");
        let gradle = jdk_bin.join("gradle.bat");
        std::fs::write(&maven, "").expect("Maven command candidate");
        std::fs::write(&gradle, "").expect("Gradle command candidate");
        assert!(!is_usable_command(&jdk_bin));
        assert!(is_usable_command(&maven));
        assert_eq!(command_path("mvn", Some(&jdk_path)), Some(maven));
        assert_eq!(command_path("gradle", Some(&jdk_path)), Some(gradle));
        std::fs::write(jdk_home.join("release"), "JAVA_VERSION=\"17.0.19\"\n")
            .expect("older JDK release metadata");
        assert_eq!(
            jdk_status_for_paths(None, Some(&jdk_path)),
            JdkStatus::Incompatible(17)
        );
    }
}
