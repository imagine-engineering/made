use std::sync::Arc;

use collections::HashMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings_macros::{MergeFrom, with_fallible_options};

#[with_fallible_options]
#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize, JsonSchema, MergeFrom)]
pub struct ExtensionSettingsContent {
    /// The extensions that should be automatically installed by Zed.
    ///
    /// This is used to make functionality provided by extensions (e.g., language support)
    /// available out-of-the-box.
    ///
    /// Default: { "html": true }
    #[serde(default)]
    pub auto_install_extensions: HashMap<Arc<str>, bool>,
    /// Failed product-default extension installs awaiting an explicit retry.
    #[serde(default)]
    pub auto_install_extension_failures: HashMap<Arc<str>, String>,
    #[serde(default)]
    pub auto_update_extensions: HashMap<Arc<str>, bool>,
    /// The capabilities granted to extensions.
    pub granted_extension_capabilities: Option<Vec<ExtensionCapabilityContent>>,
}

impl ExtensionSettingsContent {
    pub fn retry_auto_install(&mut self, extension_id: Arc<str>) {
        self.auto_install_extensions
            .insert(extension_id.clone(), true);
        self.auto_install_extension_failures.remove(&extension_id);
    }

    pub fn decline_auto_install(&mut self, extension_id: Arc<str>) {
        self.auto_install_extensions
            .insert(extension_id.clone(), false);
        self.auto_install_extension_failures.remove(&extension_id);
    }

    pub fn record_auto_install_result(&mut self, extension_id: Arc<str>, failure: Option<String>) {
        if !self
            .auto_install_extensions
            .get(&extension_id)
            .copied()
            .unwrap_or(true)
        {
            return;
        }
        if let Some(failure) = failure {
            self.auto_install_extension_failures
                .insert(extension_id, failure);
        } else {
            self.auto_install_extension_failures.remove(&extension_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExtensionSettingsContent;

    #[test]
    fn failed_install_and_retry_preserve_later_removal() {
        let mut settings = ExtensionSettingsContent::default();
        settings.auto_install_extensions.insert("java".into(), true);
        settings.record_auto_install_result("java".into(), Some("offline".into()));
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&true));
        assert_eq!(
            settings.auto_install_extension_failures.get("java"),
            Some(&"offline".into())
        );
        let saved = serde_json::to_value(&settings).expect("serialize failure state");
        settings = serde_json::from_value(saved).expect("restore failure state");
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&true));
        assert_eq!(
            settings.auto_install_extension_failures.get("java"),
            Some(&"offline".into())
        );

        settings.retry_auto_install("java".into());
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&true));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );

        settings.decline_auto_install("java".into());
        settings.record_auto_install_result("java".into(), None);
        assert_eq!(settings.auto_install_extensions.get("java"), Some(&false));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );

        settings.record_auto_install_result("java".into(), Some("late failure".into()));
        assert!(
            !settings
                .auto_install_extension_failures
                .contains_key("java")
        );
    }
}

/// A capability for an extension.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExtensionCapabilityContent {
    #[serde(rename = "process:exec")]
    ProcessExec {
        /// The command to execute.
        command: String,
        /// The arguments to pass to the command. Use `*` for a single wildcard argument.
        /// If the last element is `**`, then any trailing arguments are allowed.
        args: Vec<String>,
    },
    DownloadFile {
        host: String,
        path: Vec<String>,
    },
    #[serde(rename = "npm:install")]
    NpmInstallPackage {
        package: String,
    },
}
