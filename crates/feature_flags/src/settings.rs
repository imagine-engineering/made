use collections::HashMap;
use schemars::{Schema, json_schema};
use serde_json::{Map, Value};
use settings::{RegisterSetting, Settings, SettingsContent};

use crate::FeatureFlagStore;

#[derive(Clone, Debug, Default, RegisterSetting)]
pub struct FeatureFlagsSettings {
    pub overrides: HashMap<String, String>,
}

impl Settings for FeatureFlagsSettings {
    fn from_settings(content: &SettingsContent) -> Self {
        Self {
            overrides: content
                .feature_flags
                .as_ref()
                .map(|map| map.0.clone())
                .unwrap_or_default(),
        }
    }
}

/// Produces a JSON schema for the `feature_flags` object that lists each known
/// flag as a property with its variant keys as an `enum`.
///
/// Unknown flags are permitted via `additionalProperties: { "type": "string" }`,
/// so removing a flag from the binary never turns existing entries in
/// `settings.json` into validation errors.
pub fn generate_feature_flags_schema() -> Schema {
    let mut properties = Map::new();

    for descriptor in FeatureFlagStore::known_flags() {
        let variants = (descriptor.variants)();
        let enum_values: Vec<Value> = variants
            .iter()
            .map(|v| Value::String(v.override_key.to_string()))
            .collect();
        let enum_descriptions: Vec<Value> = variants
            .iter()
            .map(|v| Value::String(v.label.to_string()))
            .collect();

        let mut property = Map::new();
        property.insert("type".to_string(), Value::String("string".to_string()));
        property.insert("enum".to_string(), Value::Array(enum_values));
        // VS Code / json-language-server use `enumDescriptions` for hover docs
        // on each enum value; schemars passes them through untouched.
        property.insert(
            "enumDescriptions".to_string(),
            Value::Array(enum_descriptions),
        );
        let description = (descriptor.description)();
        let default_description = format!(
            "Override for the `{}` feature flag. Default: `{}` (the {} variant).",
            descriptor.name,
            (descriptor.default_variant_key)(),
            (descriptor.default_variant_key)(),
        );
        property.insert(
            "description".to_string(),
            Value::String(if description.is_empty() {
                default_description
            } else {
                let opt_in = if (descriptor.requires_local_opt_in)() {
                    " Set to `on` to opt in locally; staff status and server announcements do not enable this experiment."
                } else {
                    ""
                };
                format!("{description}{opt_in} {default_description}")
            }),
        );

        properties.insert(descriptor.name.to_string(), Value::Object(property));
    }

    json_schema!({
        "type": "object",
        "description": "Local overrides for feature flags, keyed by flag name.",
        "properties": properties,
        "additionalProperties": {
            "type": "string",
            "description": "Unknown feature flag; retained so removed flags don't trip settings validation."
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experimental_override_settings_keep_existing_persistence_and_unknown_entries() {
        let flags = serde_json::json!({
            "rust-coverage-experimental": "on",
            "rust-native-profiling-experimental": "off",
            "removed-feature": "old-variant"
        });
        let content: SettingsContent = serde_json::from_value(serde_json::json!({
            "feature_flags": flags
        }))
        .expect("existing feature flag settings wire format");
        let settings = FeatureFlagsSettings::from_settings(&content);
        assert_eq!(
            settings
                .overrides
                .get("rust-coverage-experimental")
                .map(String::as_str),
            Some("on")
        );
        assert_eq!(
            settings
                .overrides
                .get("rust-native-profiling-experimental")
                .map(String::as_str),
            Some("off")
        );
        assert_eq!(
            settings
                .overrides
                .get("removed-feature")
                .map(String::as_str),
            Some("old-variant")
        );
        let persisted = serde_json::to_value(content).expect("existing settings persistence");
        assert_eq!(persisted.get("feature_flags"), Some(&flags));
    }

    #[test]
    fn experimental_schema_exposes_descriptions_and_existing_override_wire_values() {
        let schema = generate_feature_flags_schema().to_value();
        for name in [
            "rust-coverage-experimental",
            "rust-native-profiling-experimental",
            "rust-named-cargo-presets-experimental",
            "rust-exact-doctest-actions-experimental",
        ] {
            let property = schema
                .get("properties")
                .and_then(|properties| properties.get(name))
                .expect("experimental schema property");
            assert_eq!(
                property.get("enum"),
                Some(&serde_json::json!(["on", "off"]))
            );
            assert!(
                property
                    .get("description")
                    .and_then(Value::as_str)
                    .is_some_and(|description| description.starts_with("Experimental:")
                        && description.contains("incomplete"))
            );
        }
        assert_eq!(
            schema
                .get("additionalProperties")
                .and_then(|property| property.get("type")),
            Some(&serde_json::json!("string"))
        );
        let ordinary = schema
            .get("properties")
            .and_then(|properties| properties.get("notebooks"))
            .expect("ordinary schema property");
        assert!(
            ordinary
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|description| description.starts_with("Override for"))
        );
    }
}
