use gpui::{Pixels, px};
use serde::Deserialize;
use settings::{DockSide, Settings};

pub type CargoPanelSide = DockSide;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct CargoPanelSettings {
    pub button: bool,
    pub default_width: Pixels,
    pub dock: CargoPanelSide,
    pub starts_open: bool,
}

impl Settings for CargoPanelSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        let settings = content.cargo_panel.as_ref();
        Self {
            button: settings
                .and_then(|settings| settings.button)
                .unwrap_or(true),
            default_width: px(settings
                .and_then(|settings| settings.default_width)
                .unwrap_or(280.)),
            dock: settings
                .and_then(|settings| settings.dock)
                .unwrap_or(DockSide::Right),
            starts_open: settings
                .and_then(|settings| settings.starts_open)
                .unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use settings::{CargoPanelSettingsContent, SettingsContent};

    use super::*;

    #[test]
    fn cargo_panel_settings_merge_typed_values() {
        let mut content = SettingsContent::default();
        content.cargo_panel = Some(CargoPanelSettingsContent {
            button: Some(false),
            default_width: Some(320.),
            dock: Some(DockSide::Left),
            starts_open: Some(true),
        });
        let settings = CargoPanelSettings::from_settings(&content);
        assert!(!settings.button);
        assert_eq!(settings.default_width, px(320.));
        assert_eq!(settings.dock, DockSide::Left);
        assert!(settings.starts_open);
    }
}
