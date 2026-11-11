const COLLABORATIVE_SOURCES: &[(&str, &str)] = &[
    (
        "workspace",
        include_str!("../src/collaborative_workspace.rs"),
    ),
    ("composer", include_str!("../src/collaborative_composer.rs")),
    (
        "accessibility",
        include_str!("../src/collaborative_accessibility.rs"),
    ),
    ("focus", include_str!("../src/collaborative_focus.rs")),
    ("layout", include_str!("../src/collaborative_layout.rs")),
    (
        "layout persistence",
        include_str!("../src/collaborative_layout_persistence.rs"),
    ),
    (
        "navigation",
        include_str!("../src/collaborative_navigation.rs"),
    ),
    (
        "participants",
        include_str!("../src/collaborative_participants.rs"),
    ),
    ("review", include_str!("../src/collaborative_review.rs")),
    (
        "review actions",
        include_str!("../src/collaborative_review_actions.rs"),
    ),
    (
        "review summary",
        include_str!("../src/collaborative_review_summary.rs"),
    ),
    (
        "shell state",
        include_str!("../src/collaborative_shell_state.rs"),
    ),
    ("timeline", include_str!("../src/collaborative_timeline.rs")),
    ("top bar", include_str!("../src/collaborative_top_bar.rs")),
    (
        "agent timeline",
        include_str!("../../agent_ui/src/collaborative_timeline.rs"),
    ),
    (
        "agent composer",
        include_str!("../../agent_ui/src/collaborative_composer.rs"),
    ),
    (
        "agent activity cards",
        include_str!("../../agent_ui/src/collaborative_activity_cards.rs"),
    ),
    (
        "agent settings",
        include_str!("../../agent_ui/src/collaborative_agent_settings.rs"),
    ),
    (
        "agent participants",
        include_str!("../../agent_ui/src/collaborative_participants.rs"),
    ),
    (
        "agent review",
        include_str!("../../agent_ui/src/collaborative_review.rs"),
    ),
    (
        "project review",
        include_str!("../../git_ui/src/collaborative_review.rs"),
    ),
    (
        "channel messaging",
        include_str!("../../collab_ui/src/channel_messaging.rs"),
    ),
    (
        "message timeline",
        include_str!("../../collab_ui/src/message_timeline.rs"),
    ),
    (
        "sidebar navigation",
        include_str!("../../sidebar/src/collaborative_navigation.rs"),
    ),
    (
        "sidebar pinned",
        include_str!("../../sidebar/src/collaborative_pinned.rs"),
    ),
    (
        "sidebar projects",
        include_str!("../../sidebar/src/collaborative_projects.rs"),
    ),
    (
        "sidebar rail",
        include_str!("../../sidebar/src/collaborative_rail.rs"),
    ),
    (
        "sidebar tasks",
        include_str!("../../sidebar/src/collaborative_tasks.rs"),
    ),
    (
        "visual test runner",
        include_str!("../../zed/src/visual_test_runner.rs"),
    ),
];

fn forbidden_authoritative_owner(source: &str) -> Option<&str> {
    source.lines().map(str::trim).find_map(|line| {
        let declaration = line
            .strip_prefix("pub struct ")
            .or_else(|| line.strip_prefix("pub(crate) struct "))
            .or_else(|| line.strip_prefix("struct "))?;
        let type_name = declaration
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .next()
            .unwrap_or_default();
        type_name.starts_with("Collaborative").then_some(())?;
        ["Store", "Repository", "Database", "Persistence"]
            .iter()
            .any(|suffix| type_name.ends_with(suffix))
            .then_some(type_name)
    })
}

fn declared_collaborative_types(source: &str) -> impl Iterator<Item = &str> {
    source.lines().filter_map(|line| {
        let line = line.trim();
        if line.starts_with("//") {
            return None;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        let declaration_index = match tokens.first().copied() {
            Some("struct" | "enum" | "trait" | "type") => 0,
            Some(visibility) if visibility.starts_with("pub") => 1,
            _ => return None,
        };
        let declaration = tokens.get(declaration_index).copied()?;
        if !matches!(declaration, "struct" | "enum" | "trait" | "type") {
            return None;
        }
        let type_name = tokens
            .get(declaration_index + 1)?
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .next()?;
        type_name.starts_with("Collaborative").then_some(type_name)
    })
}

#[test]
fn collaborative_modules_do_not_declare_authoritative_storage_owners() {
    for (source_name, source) in COLLABORATIVE_SOURCES {
        assert_eq!(
            forbidden_authoritative_owner(source),
            None,
            "{source_name} declares a forbidden authoritative owner"
        );
    }
}

#[test]
fn collaborative_owner_guard_rejects_a_synthetic_store() {
    assert_eq!(
        forbidden_authoritative_owner("pub struct CollaborativeMessagesStore { rows: Vec<u64> }"),
        Some("CollaborativeMessagesStore")
    );
}

#[test]
#[ignore = "requires the local collaboration ownership matrix"]
fn reuse_matrix_covers_every_declared_collaborative_type() -> std::io::Result<()> {
    let matrix = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.agents/local/collaboration/collaborative-workspace/reuse-ownership-matrix.md"
    ))?;
    for (source_name, source) in COLLABORATIVE_SOURCES {
        for type_name in declared_collaborative_types(source) {
            assert!(
                matrix.contains(&format!("`{type_name}`")),
                "{source_name} type {type_name} is missing from the reuse matrix"
            );
        }
    }
    Ok(())
}

#[test]
fn collaborative_adapters_retain_native_entity_owners() {
    let timeline = include_str!("../../agent_ui/src/collaborative_timeline.rs");
    assert!(timeline.contains("conversation_view: Entity<ConversationView>"));
    assert!(timeline.contains("collaborative_active_thread"));
    assert!(timeline.contains("render_collaborative_timeline"));

    let conversation = include_str!("../../agent_ui/src/conversation_view.rs");
    assert!(conversation.contains("threads: HashMap<acp::SessionId, Entity<ThreadView>>"));
    assert!(conversation.contains("view.render_collaborative_entries(window, cx)"));
    assert!(!conversation.contains("Vec<AgentThreadEntry>"));
    assert!(!timeline.contains("Vec<AgentThreadEntry>"));

    let composer = include_str!("../../agent_ui/src/collaborative_composer.rs");
    assert!(composer.contains("thread_view: Entity<ThreadView>"));
    assert!(composer.contains("thread_view.message_editor.clone()"));
    assert!(composer.contains("render_collaborative_composer"));
    assert!(composer.contains("MessageEditorEvent::Cancel"));

    let project_review = include_str!("../../git_ui/src/collaborative_review.rs");
    assert!(project_review.contains("project_diff: Entity<ProjectDiff>"));
    assert!(!project_review.contains("CollaborativeReviewDiffIndex"));

    let agent_review = include_str!("../../agent_ui/src/collaborative_review.rs");
    assert!(agent_review.contains("pane: Entity<AgentDiffPane>"));

    let participants = include_str!("../../agent_ui/src/collaborative_participants.rs");
    assert!(participants.contains("thread_view: WeakEntity<ThreadView>"));
    assert!(participants.contains("CollaborativeParticipantProvider::from_reader"));
    assert!(!participants.contains("view_data: CollaborativeParticipantViewData"));
}

#[test]
fn collaborative_composition_remains_multiplayer_gated() {
    let workspace_root = include_str!("../src/workspace.rs");
    let sidebar_root = include_str!("../../sidebar/src/sidebar.rs");
    assert!(
        workspace_root
            .contains("#[cfg(feature = \"multiplayer-tools\")]\npub mod collaborative_composer;")
    );
    assert!(
        sidebar_root
            .contains("#[cfg(feature = \"multiplayer-tools\")]\nmod collaborative_navigation;")
    );
}
