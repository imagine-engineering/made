pub use crate::multiplayer_capability::multiplayer_tools_available;
pub use settings::WorkspacePresentation;

pub const fn effective_workspace_presentation(
    presentation: WorkspacePresentation,
) -> WorkspacePresentation {
    if multiplayer_tools_available() {
        presentation
    } else {
        WorkspacePresentation::Editor
    }
}

pub(crate) fn serialize_workspace_presentation(
    presentation: &WorkspacePresentation,
) -> &'static str {
    match presentation {
        WorkspacePresentation::Editor => "editor",
        WorkspacePresentation::Collaborative => "collaborative",
    }
}

pub(crate) fn deserialize_workspace_presentation(
    serialized: &str,
) -> Option<WorkspacePresentation> {
    match serialized {
        "editor" => Some(WorkspacePresentation::Editor),
        "collaborative" => Some(WorkspacePresentation::Collaborative),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_presentation_setting_uses_implicit_editor_default() {
        assert_eq!(
            WorkspacePresentation::default(),
            WorkspacePresentation::Editor
        );

        let defaults: settings::UserSettingsContent =
            settings::parse_json_with_comments(&settings::default_settings())
                .expect("default settings should parse");
        assert_eq!(defaults.content.workspace.workspace_presentation, None);
    }

    #[test]
    fn unavailable_multiplayer_presentation_falls_back_without_changing_preference() {
        let preference = WorkspacePresentation::Collaborative;
        let effective = effective_workspace_presentation(preference);
        assert_eq!(preference, WorkspacePresentation::Collaborative);
        assert_eq!(
            effective,
            if cfg!(feature = "multiplayer-tools") {
                WorkspacePresentation::Collaborative
            } else {
                WorkspacePresentation::Editor
            }
        );
    }

    #[test]
    fn workspace_presentation_setting_deserializes_supported_values() {
        for (serialized, expected) in [
            ("editor", WorkspacePresentation::Editor),
            ("collaborative", WorkspacePresentation::Collaborative),
        ] {
            let content: settings::SettingsContent = settings::parse_json_with_comments(&format!(
                r#"{{"workspace_presentation":"{serialized}"}}"#
            ))
            .expect("supported workspace presentation should deserialize");
            assert_eq!(content.workspace.workspace_presentation, Some(expected));
        }

        let invalid: Result<settings::SettingsContent, _> =
            settings::parse_json_with_comments(r#"{"workspace_presentation":"unknown"}"#);
        assert!(invalid.is_err());
    }

    #[test]
    fn workspace_presentation_setting_schema_lists_supported_values() {
        let schema =
            serde_json::to_value(schemars::schema_for!(settings::WorkspaceSettingsContent))
                .expect("workspace settings schema should serialize");
        let schema = schema.to_string();
        assert!(schema.contains("workspace_presentation"));
        assert!(schema.contains("editor"));
        assert!(schema.contains("collaborative"));
    }
}
