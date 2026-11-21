//! Focused tests for the dependency analyzer item: conflicts/unresolved
//! filters, the GroupId toggle, tree/list parity over the same admitted
//! occurrence identities, Usages pane updates on selection and state
//! retention across tab switches.

use crate::dependency_analyzer_item::{
    DependencyAnalyzerDerivation, DependencyAnalyzerFilter, DependencyAnalyzerGraph,
    DependencyAnalyzerItem, DependencyAnalyzerPresentation, analyzer_row_id_for_node,
    analyzer_rows, analyzer_usages,
};
use crate::dependency_diagram_item::{
    DependencyDiagramProvider, DependencyDiagramScope, DependencyDiagramTarget,
};
use gpui::TestAppContext;
use language_tools::language_tool_tree::LanguageToolNode;
use project::jvm_project::{
    GradleDependencyComponent, GradleDependencyEdge, GradleDependencyKind,
    GradleDependencySelector, MavenDependencyComponent, MavenDependencyEdge, MavenDependencyNode,
    MavenUnavailableDescriptor,
};
use settings::SettingsStore;
use theme::LoadThemes;
use workspace::Item as _;

fn gradle_component(
    id: &str,
    group: Option<&str>,
    module: Option<&str>,
    version: Option<&str>,
    conflict_resolution: bool,
) -> GradleDependencyComponent {
    GradleDependencyComponent {
        id: id.into(),
        kind: GradleDependencyKind::Module,
        group: group.map(|group| group.into()),
        module: module.map(|module| module.into()),
        version: version.map(|version| version.into()),
        build_tree_path: None,
        project_path: None,
        conflict_resolution,
        forced: false,
        selected_by_rule: false,
        composite_substitution: false,
        constrained: false,
        selection_causes: Vec::new(),
    }
}

fn gradle_module_selector(group: &str, module: &str, version: &str) -> GradleDependencySelector {
    GradleDependencySelector::Module {
        group: group.into(),
        module: module.into(),
        version: version.into(),
        required_version: String::new(),
        preferred_version: String::new(),
        strict_version: String::new(),
        rejected_versions: Vec::new(),
    }
}

fn gradle_edge(
    from: &str,
    selected: Option<&str>,
    resolved: bool,
    requested: GradleDependencySelector,
) -> GradleDependencyEdge {
    GradleDependencyEdge {
        from: from.into(),
        requested,
        constraint: false,
        origin_known: true,
        resolved,
        selected: selected.map(|selected| selected.into()),
        selected_variant: None,
        failure_category: None,
    }
}

fn gradle_graph() -> DependencyAnalyzerGraph {
    let components = vec![
        gradle_component("root", None, None, None, false),
        gradle_component("left", Some("org.example"), Some("left"), Some("1.0"), true),
        gradle_component(
            "right",
            Some("org.example"),
            Some("right"),
            Some("2.0"),
            false,
        ),
        gradle_component(
            "leaf",
            Some("org.example"),
            Some("leaf"),
            Some("3.0"),
            false,
        ),
    ];
    let edges = vec![
        gradle_edge(
            "root",
            Some("left"),
            true,
            gradle_module_selector("org.example", "left", "1.0"),
        ),
        gradle_edge(
            "root",
            Some("right"),
            true,
            gradle_module_selector("org.example", "right", "2.0"),
        ),
        gradle_edge(
            "left",
            Some("leaf"),
            true,
            gradle_module_selector("org.example", "leaf", "3.0"),
        ),
        gradle_edge(
            "root",
            None,
            false,
            gradle_module_selector("org.example", "missing", "9.9"),
        ),
    ];
    DependencyAnalyzerGraph::Gradle {
        configuration: ":core".into(),
        components,
        edges,
    }
}

fn maven_component(
    id: &str,
    group_id: &str,
    artifact_id: &str,
    version: &str,
) -> MavenDependencyComponent {
    MavenDependencyComponent {
        id: id.into(),
        group_id: group_id.into(),
        artifact_id: artifact_id.into(),
        extension: "jar".into(),
        classifier: String::new(),
        version: version.into(),
    }
}

fn maven_node(id: &str, component: &str) -> MavenDependencyNode {
    MavenDependencyNode {
        id: id.into(),
        component: component.into(),
    }
}

fn maven_edge(
    from_node: &str,
    target_node: &str,
    selected_node: &str,
    scope: &str,
    omitted: bool,
    position: usize,
) -> MavenDependencyEdge {
    MavenDependencyEdge {
        from: "c-root".into(),
        from_node: from_node.into(),
        target_node: target_node.into(),
        selected_node: selected_node.into(),
        position,
        component: "c-root".into(),
        selected: "c-root".into(),
        omitted,
        direct: true,
        version_constraint: None,
        premanaged_version: None,
        premanaged_scope: None,
        premanaged_optional: None,
        managed_bits: 0,
        original_scope: None,
        original_optional: None,
        scope: scope.into(),
        optional: false,
        exclusions: Vec::new(),
        premanaged_exclusions: None,
    }
}

fn maven_graph() -> DependencyAnalyzerGraph {
    let components = vec![
        maven_component("c-root", "org.example", "app", "1.0"),
        maven_component("c-a", "org.example", "a", "1.0"),
        maven_component("c-b", "org.example", "b", "2.0"),
        maven_component("c-leaf", "org.example", "leaf", "3.0"),
    ];
    let nodes = vec![
        maven_node("n-root", "c-root"),
        maven_node("n-a", "c-a"),
        maven_node("n-b", "c-b"),
        maven_node("n-leaf", "c-leaf"),
    ];
    let edges = vec![
        maven_edge("n-root", "n-a", "n-a", "compile", false, 0),
        maven_edge("n-root", "n-b", "n-b", "runtime", true, 1),
        maven_edge("n-a", "n-leaf", "n-leaf", "test", false, 2),
    ];
    let unavailable_descriptors = vec![MavenUnavailableDescriptor {
        component: "c-leaf".into(),
        problem: "Descriptor unavailable".into(),
    }];
    DependencyAnalyzerGraph::Maven {
        nodes,
        components,
        edges,
        unavailable_descriptors,
    }
}

fn node_keys(rows: &[LanguageToolNode]) -> Vec<String> {
    let mut keys = Vec::new();
    for row in rows {
        if let Some(key) = row.id.0.strip_prefix("analyzer-node:") {
            keys.push(key.to_owned());
        }
        keys.extend(node_keys(&row.children));
    }
    keys
}

fn row_labels(rows: &[LanguageToolNode]) -> Vec<String> {
    let mut labels = Vec::new();
    for row in rows {
        labels.push(row.label.clone());
        labels.extend(row_labels(&row.children));
    }
    labels
}

#[test]
fn analyzer_rows_filters_group_ids_and_presentation_parity() {
    let graph = gradle_graph();
    let tree = analyzer_rows(
        &graph,
        DependencyAnalyzerFilter::All,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    let labels = row_labels(&tree);
    assert!(
        labels.iter().any(|label| label == "org.example"),
        "group rows carry the group ID: {labels:?}"
    );
    assert!(
        labels.iter().any(|label| label == "org.example:left"),
        "artifact rows expose the module coordinate: {labels:?}"
    );
    assert!(
        labels.iter().any(|label| label == "1.0"),
        "occurrence rows expose the version: {labels:?}"
    );
    assert!(
        labels
            .iter()
            .any(|label| label.contains("missing") && label.contains("9.9")),
        "unresolved requests stay explicit placeholder rows: {labels:?}"
    );

    // Tree and flat presentations report the same occurrence identities.
    let flat = analyzer_rows(
        &graph,
        DependencyAnalyzerFilter::All,
        DependencyAnalyzerPresentation::Flat,
        true,
    );
    let mut tree_keys = node_keys(&tree);
    let mut flat_keys = node_keys(&flat);
    tree_keys.sort();
    flat_keys.sort();
    assert_eq!(
        tree_keys, flat_keys,
        "tree and flat presentations share occurrence identities"
    );
    assert_eq!(
        tree_keys.len(),
        5,
        "root, three resolved and one unresolved"
    );
    assert!(tree_keys.contains(&"left".to_owned()));
    assert!(tree_keys.contains(&"unresolved:org.example:missing:9.9".to_owned()));

    // The conflicts filter keeps only the conflicted subtree.
    let conflicts = analyzer_rows(
        &graph,
        DependencyAnalyzerFilter::Conflicts,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    assert_eq!(
        node_keys(&conflicts),
        ["left".to_owned()],
        "only the conflicted component survives"
    );

    // The unresolved filter keeps only unresolved placeholders.
    let unresolved = analyzer_rows(
        &graph,
        DependencyAnalyzerFilter::Unresolved,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    assert_eq!(
        node_keys(&unresolved),
        ["unresolved:org.example:missing:9.9".to_owned()]
    );

    // The GroupId toggle changes flat labels without changing identities.
    let flat_off = analyzer_rows(
        &graph,
        DependencyAnalyzerFilter::All,
        DependencyAnalyzerPresentation::Flat,
        false,
    );
    let labels_on = row_labels(&flat);
    let labels_off = row_labels(&flat_off);
    assert!(
        labels_on
            .iter()
            .any(|label| label == "org.example:left:1.0"),
        "group IDs are shown by default: {labels_on:?}"
    );
    assert!(
        labels_off.iter().any(|label| label == "left:1.0"),
        "group IDs are hidden after the toggle: {labels_off:?}"
    );
    let mut flat_off_keys = node_keys(&flat_off);
    flat_off_keys.sort();
    assert_eq!(
        flat_off_keys, flat_keys,
        "the toggle keeps identities stable"
    );

    // Maven rows expose group, artifact, version and scope, with the same
    // conflict and unresolved filters.
    let maven = maven_graph();
    let maven_tree = analyzer_rows(
        &maven,
        DependencyAnalyzerFilter::All,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    let maven_labels = row_labels(&maven_tree);
    assert!(maven_labels.iter().any(|label| label == "org.example"));
    assert!(maven_labels.iter().any(|label| label == "org.example:a"));
    assert!(
        maven_labels
            .iter()
            .any(|label| label == "1.0 — compile" || label == "2.0 — runtime"),
        "occurrence rows carry version and scope: {maven_labels:?}"
    );
    let maven_conflicts = analyzer_rows(
        &maven,
        DependencyAnalyzerFilter::Conflicts,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    assert_eq!(node_keys(&maven_conflicts), ["n-b".to_owned()]);
    let maven_unresolved = analyzer_rows(
        &maven,
        DependencyAnalyzerFilter::Unresolved,
        DependencyAnalyzerPresentation::Tree,
        true,
    );
    assert_eq!(node_keys(&maven_unresolved), ["n-leaf".to_owned()]);
    let maven_tree_off = analyzer_rows(
        &maven,
        DependencyAnalyzerFilter::All,
        DependencyAnalyzerPresentation::Tree,
        false,
    );
    assert!(
        row_labels(&maven_tree_off).iter().any(|label| label == "a"),
        "Maven tree artifact labels drop the group ID after the toggle"
    );
}

#[test]
fn analyzer_usages_report_admitted_reverse_edges() {
    let labels = |rows: &[crate::dependency_analyzer_item::AnalyzerUsage]| {
        rows.iter()
            .map(|usage| usage.label.clone())
            .collect::<Vec<_>>()
    };
    let gradle = gradle_graph();
    let usages = analyzer_usages(&gradle, "leaf");
    assert_eq!(
        labels(&usages.rows),
        ["used by org.example:left".to_owned()]
    );
    assert!(!usages.truncated);
    assert!(
        usages.rows.iter().all(|usage| usage.declaration.is_none()),
        "the fixture's declaring components report no project path, so no declaration is trustworthy"
    );
    assert!(
        analyzer_usages(&gradle, "root").rows.is_empty(),
        "the root has no admitted users"
    );
    assert!(
        analyzer_usages(&gradle, "unresolved:org.example:missing:9.9")
            .rows
            .is_empty(),
        "unresolved placeholders have no admitted users"
    );

    let maven = maven_graph();
    let usages = analyzer_usages(&maven, "n-leaf");
    assert_eq!(
        labels(&usages.rows),
        ["used by org.example:a; scope test".to_owned()]
    );
    assert!(
        usages.rows.iter().all(|usage| usage.declaration.is_none()),
        "Maven facts carry no Gradle declaration"
    );
    let usages = analyzer_usages(&maven, "n-b");
    assert_eq!(
        labels(&usages.rows),
        ["used by org.example:app; scope runtime; omitted by mediation".to_owned()],
        "mediation facts stay explicit in the usage rows"
    );

    // The pane is bounded with an explicit truncation count.
    let mut components = vec![maven_component("c-sel", "org.example", "sel", "1.0")];
    let mut nodes = vec![maven_node("n-sel", "c-sel")];
    let mut edges = Vec::new();
    for index in 0..70 {
        let user = format!("c-u{index}");
        let node = format!("n-u{index}");
        nodes.push(maven_node(&node, &user));
        components.push(maven_component(
            &user,
            "org.example",
            &format!("u{index}"),
            "1.0",
        ));
        let edge = maven_edge(&node, "n-sel", "n-sel", "compile", false, index);
        edges.push(MavenDependencyEdge { from: user, ..edge });
    }
    let graph = DependencyAnalyzerGraph::Maven {
        nodes,
        components,
        edges,
        unavailable_descriptors: Vec::new(),
    };
    let usages = analyzer_usages(&graph, "n-sel");
    assert_eq!(usages.rows.len(), 64, "usages are bounded at 64 rows");
    assert!(usages.truncated, "truncation stays explicit");
}

#[test]
fn analyzer_menu_targets_rows_and_usages() {
    use crate::dependency_analyzer_item::analyzer_declaration_target;
    use language_tools::language_tool_tree::LanguageToolNodeId;

    // A Gradle graph whose declaring component reports its project path.
    let mut components = vec![
        gradle_component("root", None, None, None, false),
        gradle_component(
            "leaf",
            Some("org.example"),
            Some("leaf"),
            Some("3.0"),
            false,
        ),
    ];
    components[0].project_path = Some(":declared".into());
    let graph = DependencyAnalyzerGraph::Gradle {
        configuration: "runtimeClasspath".into(),
        components,
        edges: vec![gradle_edge(
            "root",
            Some("leaf"),
            true,
            gradle_module_selector("org.example", "leaf", "3.0"),
        )],
    };
    assert_eq!(
        analyzer_declaration_target(&graph, "leaf"),
        Some(":declared".into()),
        "the admitted reverse edge with a known origin names the declaring module"
    );
    assert_eq!(
        analyzer_declaration_target(&graph, "root"),
        None,
        "the root has no admitted reverse edge"
    );

    // An origin the provider does not report as known is not trustworthy.
    let mut unknown = gradle_edge(
        "root",
        Some("leaf"),
        true,
        gradle_module_selector("org.example", "leaf", "3.0"),
    );
    unknown.origin_known = false;
    let mut components = vec![
        gradle_component("root", None, None, None, false),
        gradle_component(
            "leaf",
            Some("org.example"),
            Some("leaf"),
            Some("3.0"),
            false,
        ),
    ];
    components[0].project_path = Some(":declared".into());
    let graph = DependencyAnalyzerGraph::Gradle {
        configuration: "runtimeClasspath".into(),
        components,
        edges: vec![unknown],
    };
    assert_eq!(
        analyzer_declaration_target(&graph, "leaf"),
        None,
        "an unknown edge origin offers no declaration navigation"
    );

    // Maven facts carry no Gradle declaration, and wrapper rows carry no node
    // target at all.
    let maven = maven_graph();
    assert_eq!(
        analyzer_declaration_target(&maven, "n-leaf"),
        None,
        "Maven rows offer no Gradle declaration"
    );
    assert!(
        crate::dependency_analyzer_item::analyzer_node_key(&LanguageToolNodeId(
            "analyzer-group:org.example".into()
        ))
        .is_none(),
        "group wrapper rows target no node"
    );
}

fn init(cx: &mut gpui::App) {
    let settings = SettingsStore::test(cx);
    cx.set_global(settings);
    theme_settings::init(LoadThemes::JustBase, cx);
    editor::init(cx);
    crate::init(cx);
}

#[gpui::test]
async fn analyzer_item_filters_usages_and_state_survive_tab_switches(cx: &mut TestAppContext) {
    cx.update(|cx| init(cx));
    let fs = project::FakeFs::new(cx.executor());
    fs.insert_tree(
        "/groovy-analyzer",
        serde_json::json!({"pom.xml": "<project/>"}),
    )
    .await;
    let project = project::Project::test(fs, ["/groovy-analyzer".as_ref()], cx).await;
    let (container, cx) = cx.add_window_view(|window, cx| {
        workspace::MultiWorkspace::test_new(project.clone(), window, cx)
    });
    let workspace = container.read_with(cx, |container, _| container.workspace().clone());
    let derivation = DependencyAnalyzerDerivation {
        target: DependencyDiagramTarget {
            module_label: "app".into(),
            provider: DependencyDiagramProvider::Gradle,
            configuration: Some(":core".into()),
        },
        graph: gradle_graph(),
        scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
        scope_options: vec![":core".into()],
    };
    let item = workspace.update_in(cx, |workspace, window, cx| {
        DependencyAnalyzerItem::open(derivation, None, workspace, window, cx)
    });
    let (text, options) = cx.update(|_, cx| {
        (
            item.read(cx).tab_content_text(0, cx).to_string(),
            item.read(cx).scope_options().to_vec(),
        )
    });
    assert_eq!(text, "Dependency Analyzer — app (:core)");
    assert_eq!(options, [":core".to_owned()]);

    // The Usages pane updates on selection; the tree starts collapsed so the
    // test expands every admitted row first, like the toolbar's Expand all.
    item.update(cx, |item, cx| {
        item.expand_all_rows(cx);
        item.select_node_row(analyzer_row_id_for_node("leaf"), cx)
    });
    let usages = cx.update(|_, cx| {
        item.read(cx)
            .usages()
            .iter()
            .map(|usage| usage.label.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(usages, ["used by org.example:left".to_owned()]);

    // Search lists admitted rows and selecting a match selects and reveals it.
    let matches = cx.update(|_, cx| item.read(cx).search_matches(""));
    assert!(!matches.is_empty(), "admitted rows are searchable");
    item.update(cx, |item, cx| item.reveal_row("1.0", cx));
    let (selected_key, usages) = cx.update(|_, cx| {
        (
            item.read(cx).selected_key().map(|key| key.to_owned()),
            item.read(cx)
                .usages()
                .iter()
                .map(|usage| usage.label.clone())
                .collect::<Vec<_>>(),
        )
    });
    assert_eq!(selected_key.as_deref(), Some("left"));
    assert_eq!(usages, ["used by root".to_owned()]);

    // The conflicts filter retains the conflicted selection and its usages.
    item.update(cx, |item, cx| {
        item.apply_filter(DependencyAnalyzerFilter::Conflicts, cx)
    });
    let (filter, selected_key, usages) = cx.update(|_, cx| {
        (
            item.read(cx).filter(),
            item.read(cx).selected_key().map(|key| key.to_owned()),
            item.read(cx)
                .usages()
                .iter()
                .map(|usage| usage.label.clone())
                .collect::<Vec<_>>(),
        )
    });
    assert_eq!(filter, DependencyAnalyzerFilter::Conflicts);
    assert_eq!(selected_key.as_deref(), Some("left"));
    assert_eq!(usages, ["used by root".to_owned()]);

    // The unresolved filter drops a selection the filtered rows cannot
    // represent and clears the Usages pane with it.
    item.update(cx, |item, cx| {
        item.apply_filter(DependencyAnalyzerFilter::Unresolved, cx)
    });
    let (selected_key, usages_empty) = cx.update(|_, cx| {
        (
            item.read(cx).selected_key().map(|key| key.to_owned()),
            item.read(cx).usages().is_empty(),
        )
    });
    assert_eq!(selected_key, None);
    assert!(usages_empty);

    // GroupId toggle, presentation switch and the restored filter stay on the
    // item across renders.
    item.update(cx, |item, cx| {
        item.toggle_group_ids(cx);
        item.toggle_presentation(cx);
        item.apply_filter(DependencyAnalyzerFilter::All, cx);
    });
    let (show_group_ids, presentation) =
        cx.update(|_, cx| (item.read(cx).show_group_ids(), item.read(cx).presentation()));
    assert!(!show_group_ids);
    assert_eq!(presentation, DependencyAnalyzerPresentation::Flat);

    // Tab switch: opening a second analyzer tab and coming back preserves
    // the first item's state.
    let second = DependencyAnalyzerDerivation {
        target: DependencyDiagramTarget {
            module_label: "other".into(),
            provider: DependencyDiagramProvider::Gradle,
            configuration: Some(":core".into()),
        },
        graph: gradle_graph(),
        scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
        scope_options: vec![":core".into()],
    };
    let _second = workspace.update_in(cx, |workspace, window, cx| {
        DependencyAnalyzerItem::open(second, None, workspace, window, cx)
    });
    workspace.update_in(cx, |workspace, window, cx| {
        workspace.activate_item(&item, true, true, window, cx)
    });
    let (show_group_ids, presentation, filter) = cx.update(|_, cx| {
        (
            item.read(cx).show_group_ids(),
            item.read(cx).presentation(),
            item.read(cx).filter(),
        )
    });
    assert!(!show_group_ids, "GroupId toggle survives the tab switch");
    assert_eq!(presentation, DependencyAnalyzerPresentation::Flat);
    assert_eq!(filter, DependencyAnalyzerFilter::All);
}

#[gpui::test]
async fn analyzer_row_menu_deploys_targets_and_dismisses(cx: &mut TestAppContext) {
    cx.update(|cx| init(cx));
    let fs = project::FakeFs::new(cx.executor());
    fs.insert_tree(
        "/groovy-analyzer-menu",
        serde_json::json!({"pom.xml": "<project/>"}),
    )
    .await;
    let project = project::Project::test(fs, ["/groovy-analyzer-menu".as_ref()], cx).await;
    let (container, cx) = cx.add_window_view(|window, cx| {
        workspace::MultiWorkspace::test_new(project.clone(), window, cx)
    });
    let workspace = container.read_with(cx, |container, _| container.workspace().clone());
    let derivation = DependencyAnalyzerDerivation {
        target: DependencyDiagramTarget {
            module_label: "app".into(),
            provider: DependencyDiagramProvider::Gradle,
            configuration: Some(":core".into()),
        },
        graph: gradle_graph(),
        scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
        scope_options: vec![":core".into()],
    };
    let item = workspace.update_in(cx, |workspace, window, cx| {
        DependencyAnalyzerItem::open(derivation, None, workspace, window, cx)
    });
    item.update(cx, |item, cx| {
        item.expand_all_rows(cx);
        item.select_node_row(analyzer_row_id_for_node("leaf"), cx)
    });

    // Staging the row menu deploys it anchored at the given position.
    cx.update(|window, cx| {
        item.update(cx, |item, cx| {
            item.show_row_context_menu(
                analyzer_row_id_for_node("leaf"),
                gpui::point(gpui::px(40.0), gpui::px(40.0)),
                window,
                cx,
            );
        });
    });
    assert!(
        cx.update(|_, cx| item.read(cx).row_context_menu_is_deployed()),
        "the row context menu is deployed"
    );

    // Moving focus off the menu dismisses it: the ui::ContextMenu cancels on
    // blur and emits DismissEvent, which takes the staged menu.
    use gpui::Focusable as _;
    cx.update(|window, cx| {
        item.update(cx, |item, cx| {
            let handle = item.focus_handle(cx);
            window.focus(&handle, cx);
        });
    });
    assert!(
        !cx.update(|_, cx| item.read(cx).row_context_menu_is_deployed()),
        "focusing away dismisses the row context menu"
    );

    // The menu commands route through the opening panel; without one they
    // report guidance instead of dispatching.
    let notice = cx.update(|window, cx| {
        item.update(cx, |item, cx| {
            item.go_to_dependency_declaration(":declared", window, cx);
            item.notice().map(|notice| notice.to_string())
        })
    });
    assert!(
        notice
            .as_deref()
            .is_some_and(|notice| notice.contains("opening panel is unavailable")),
        "Go to Gradle Declaration without an opening panel reports guidance: {notice:?}"
    );
    let notice = cx.update(|_, cx| {
        item.update(cx, |item, cx| {
            item.show_in_dependency_diagram("leaf", cx);
            item.notice().map(|notice| notice.to_string())
        })
    });
    assert!(
        notice
            .as_deref()
            .is_some_and(|notice| notice.contains("opening panel is unavailable")),
        "Show in Diagram without an opening panel reports guidance: {notice:?}"
    );

    // A Usages pane row stages the same menu targeting its specific usage.
    item.update(cx, |item, cx| {
        item.select_node_row(analyzer_row_id_for_node("leaf"), cx)
    });
    cx.update(|window, cx| {
        item.update(cx, |item, cx| {
            item.show_usage_context_menu(
                0,
                gpui::point(gpui::px(60.0), gpui::px(60.0)),
                window,
                cx,
            );
        });
    });
    assert!(
        cx.update(|_, cx| item.read(cx).row_context_menu_is_deployed()),
        "the usage row context menu is deployed"
    );
}

#[gpui::test]
async fn analyzer_diagram_linkage_preselects_nodes_and_rejects_untrusted(cx: &mut TestAppContext) {
    cx.update(|cx| init(cx));
    let fs = project::FakeFs::new(cx.executor());
    fs.insert_tree(
        "/groovy-analyzer-linkage",
        serde_json::json!({"pom.xml": "<project/>"}),
    )
    .await;
    let project = project::Project::test(fs, ["/groovy-analyzer-linkage".as_ref()], cx).await;
    let (container, cx) = cx.add_window_view(|window, cx| {
        workspace::MultiWorkspace::test_new(project.clone(), window, cx)
    });
    let workspace = container.read_with(cx, |container, _| container.workspace().clone());
    let derivation = DependencyAnalyzerDerivation {
        target: DependencyDiagramTarget {
            module_label: "app".into(),
            provider: DependencyDiagramProvider::Gradle,
            configuration: Some(":core".into()),
        },
        graph: gradle_graph(),
        scope: DependencyDiagramScope::GradleConfiguration(":core".into()),
        scope_options: vec![":core".into()],
    };
    let item = workspace.update_in(cx, |workspace, window, cx| {
        DependencyAnalyzerItem::open(derivation, None, workspace, window, cx)
    });
    item.update(cx, |item, cx| item.expand_all_rows(cx));

    // Preselecting the diagram's node carries the selection and its Usages
    // pane over, like All usages from the diagram promises.
    item.update(cx, |item, cx| item.preselect_node("leaf", cx));
    let (selected_key, usages) = cx.update(|_, cx| {
        (
            item.read(cx).selected_key().map(|key| key.to_owned()),
            item.read(cx)
                .usages()
                .iter()
                .map(|usage| usage.label.clone())
                .collect::<Vec<_>>(),
        )
    });
    assert_eq!(selected_key.as_deref(), Some("leaf"));
    assert_eq!(usages, ["used by org.example:left".to_owned()]);

    // A key the current rows cannot represent leaves the selection untouched,
    // so a diagram node outside the admitted facts preselects nothing.
    item.update(cx, |item, cx| item.preselect_node("not-admitted", cx));
    let selected_key = cx.update(|_, cx| item.read(cx).selected_key().map(|key| key.to_owned()));
    assert_eq!(selected_key.as_deref(), Some("leaf"));

    // The untrusted-location rejection: an edge whose origin the provider
    // does not report as known offers no Go to Gradle Declaration target.
    use crate::dependency_analyzer_item::analyzer_declaration_target;
    let mut unknown = gradle_edge(
        "root",
        Some("leaf"),
        true,
        gradle_module_selector("org.example", "leaf", "3.0"),
    );
    unknown.origin_known = false;
    let mut components = vec![
        gradle_component("root", None, None, None, false),
        gradle_component(
            "leaf",
            Some("org.example"),
            Some("leaf"),
            Some("3.0"),
            false,
        ),
    ];
    components[0].project_path = Some(":declared".into());
    let graph = DependencyAnalyzerGraph::Gradle {
        configuration: "runtimeClasspath".into(),
        components,
        edges: vec![unknown],
    };
    assert_eq!(
        analyzer_declaration_target(&graph, "leaf"),
        None,
        "an untrusted source location offers no declaration navigation"
    );
}
