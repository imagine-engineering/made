use std::{path::PathBuf, sync::Arc};

use cargo_ui::{
    CargoAction, CargoActionDispatcher, CargoActionNodeKind, CargoActionPlan, CargoActionSelection,
    CargoActionTargetKind, CargoPreset, CargoTargetSelector, dispatch_cargo_action,
    plan_cargo_action,
};
use gpui::TestAppContext;
use project::{
    Inventory, ProjectPath, TaskSourceKind, WorktreeId, task_store::TaskSettingsLocation,
};
use settings::SettingsLocation;
use task::{DebugScenario, SharedTaskContext, TaskContext, TaskTemplate};
use util::rel_path::RelPath;

const GLOBAL_TASKS: &str = r#"[
  {"label":"Global check","command":"global-check","args":["--global"]}
]"#;
const PROJECT_TASKS: &str = r#"[
  {"label":"Project check","command":"project-check","args":["--project"]}
]"#;
const GLOBAL_DEBUG: &str = r#"[
  {"label":"Global debug","adapter":"CodeLLDB","build":{"command":"global-build","args":["--global"]}}
]"#;
const PROJECT_DEBUG: &str = r#"[
  {"label":"Project debug","adapter":"CodeLLDB","build":{"command":"project-build","args":["--project"]}}
]"#;

#[derive(Default)]
struct RecordingDispatcher {
    tasks: Vec<(TaskSourceKind, TaskTemplate, TaskContext, bool)>,
    debug: Vec<(DebugScenario, TaskContext, WorktreeId)>,
    pre_launch: Vec<String>,
    native_profile: Vec<(CargoActionSelection, CargoPreset, TaskContext)>,
}

impl CargoActionDispatcher for RecordingDispatcher {
    fn schedule_native_profile(
        &mut self,
        selection: CargoActionSelection,
        preset: CargoPreset,
        context: TaskContext,
    ) {
        self.native_profile.push((selection, preset, context));
    }

    fn schedule_task(
        &mut self,
        source: TaskSourceKind,
        template: TaskTemplate,
        context: TaskContext,
        omit_history: bool,
    ) {
        self.tasks.push((source, template, context, omit_history));
    }

    fn start_debug_session(
        &mut self,
        scenario: DebugScenario,
        context: TaskContext,
        worktree_id: WorktreeId,
    ) {
        self.debug.push((scenario, context, worktree_id));
    }

    fn run_pre_launch_task(
        &mut self,
        reference: String,
        _context: TaskContext,
        _worktree_id: WorktreeId,
        _continuation: CargoActionPlan,
    ) {
        self.pre_launch.push(reference);
    }
}

fn selection(worktree_id: WorktreeId) -> CargoActionSelection {
    CargoActionSelection {
        node_kind: CargoActionNodeKind::Target(CargoActionTargetKind::Binary),
        worktree_id,
        workspace_name: "workspace".to_string(),
        workspace_manifest: ProjectPath {
            worktree_id,
            path: Arc::from(
                RelPath::from_unix_str("Cargo.toml").expect("fixture path should be valid"),
            ),
        },
        package_name: Some("member".to_string()),
        package_manifest: Some(ProjectPath {
            worktree_id,
            path: Arc::from(
                RelPath::from_unix_str("member/Cargo.toml").expect("fixture path should be valid"),
            ),
        }),
        target: Some(CargoTargetSelector::Binary("member".to_string())),
        has_bench_targets: false,
    }
}

#[gpui::test]
async fn tasks_json_and_debug_json_coexist_with_cargo_configuration(cx: &mut TestAppContext) {
    let worktree_id = WorktreeId::from_proto(17);
    let project_root = RelPath::from_unix_str("").expect("empty project root should be valid");
    let global_tasks_path = PathBuf::from("/config/tasks.json");
    let global_debug_path = PathBuf::from("/config/debug.json");
    let inventory = cx.update(|cx| Inventory::new(cx));
    let raw_before = (
        GLOBAL_TASKS.to_string(),
        PROJECT_TASKS.to_string(),
        GLOBAL_DEBUG.to_string(),
        PROJECT_DEBUG.to_string(),
    );
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(&global_tasks_path),
                Some(GLOBAL_TASKS),
            )
            .expect("global tasks should load");
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id,
                    path: &project_root,
                }),
                Some(PROJECT_TASKS),
            )
            .expect("project tasks should load");
        inventory
            .update_file_based_scenarios(
                TaskSettingsLocation::Global(&global_debug_path),
                Some(GLOBAL_DEBUG),
            )
            .expect("global debug scenarios should load");
        inventory
            .update_file_based_scenarios(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id,
                    path: &project_root,
                }),
                Some(PROJECT_DEBUG),
            )
            .expect("project debug scenarios should load");
    });

    let task_context = TaskContext {
        cwd: Some(PathBuf::from("/remote/workspace")),
        ..TaskContext::default()
    };
    let before_tasks = inventory
        .read_with(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(worktree_id), cx)
        })
        .await;
    assert_eq!(before_tasks[0].1.label, "Project check");
    assert_eq!(before_tasks[1].1.label, "Global check");
    let before_debug = inventory
        .update(cx, |inventory, cx| {
            inventory.list_debug_scenarios(
                &project::TaskContexts {
                    active_worktree_context: Some((worktree_id, task_context.clone())),
                    ..project::TaskContexts::default()
                },
                Vec::new(),
                Vec::new(),
                false,
                cx,
            )
        })
        .await
        .1;
    assert_eq!(before_debug.len(), 2);

    let user_resolved = before_tasks[0]
        .1
        .resolve_task(&before_tasks[0].0.to_id_base(), &task_context)
        .expect("project task should resolve");
    inventory.update(cx, |inventory, _| {
        inventory.task_scheduled(before_tasks[0].0.clone(), user_resolved.clone());
        inventory.scenario_scheduled(
            before_debug[0].1.clone(),
            SharedTaskContext::from(task_context.clone()),
            Some(worktree_id),
            None,
        );
    });

    let cargo_run = plan_cargo_action(
        CargoAction::Run,
        &selection(worktree_id),
        None,
        &task_context,
    )
    .expect("Cargo run should plan");
    let cargo_debug = plan_cargo_action(
        CargoAction::Debug,
        &selection(worktree_id),
        None,
        &task_context,
    )
    .expect("Cargo debug should plan");
    let mut dispatcher = RecordingDispatcher::default();
    dispatch_cargo_action(cargo_run, &mut dispatcher);
    dispatch_cargo_action(cargo_debug, &mut dispatcher);
    assert_eq!(dispatcher.tasks.len(), 1);
    assert_eq!(dispatcher.debug.len(), 1);
    assert!(dispatcher.pre_launch.is_empty());
    let cargo_resolved = dispatcher.tasks[0]
        .1
        .resolve_task("cargo-compatibility", &dispatcher.tasks[0].2)
        .expect("Cargo task should resolve");
    inventory.update(cx, |inventory, _| {
        inventory.task_scheduled(dispatcher.tasks[0].0.clone(), cargo_resolved.clone());
        inventory.scenario_scheduled(
            dispatcher.debug[0].0.clone(),
            SharedTaskContext::from(dispatcher.debug[0].1.clone()),
            Some(worktree_id),
            None,
        );
    });

    let after_tasks = inventory
        .read_with(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(worktree_id), cx)
        })
        .await;
    let after_debug = inventory
        .update(cx, |inventory, cx| {
            inventory.list_debug_scenarios(
                &project::TaskContexts {
                    active_worktree_context: Some((worktree_id, task_context.clone())),
                    ..project::TaskContexts::default()
                },
                Vec::new(),
                Vec::new(),
                false,
                cx,
            )
        })
        .await;

    assert_eq!(before_tasks, after_tasks);
    assert_eq!(before_debug, after_debug.1);
    assert_eq!(GLOBAL_TASKS, raw_before.0);
    assert_eq!(PROJECT_TASKS, raw_before.1);
    assert_eq!(GLOBAL_DEBUG, raw_before.2);
    assert_eq!(PROJECT_DEBUG, raw_before.3);
    cx.update(|cx| {
        assert!(
            inventory
                .read(cx)
                .last_scheduled_task(Some(&user_resolved.id))
                .is_some()
        );
        assert!(
            inventory
                .read(cx)
                .last_scheduled_task(Some(&cargo_resolved.id))
                .is_some()
        );
    });
    let scheduled_debug_labels = after_debug
        .0
        .iter()
        .map(|(scenario, _)| scenario.label.as_ref())
        .collect::<Vec<_>>();
    assert!(scheduled_debug_labels.contains(&"Project debug"));
    assert!(
        scheduled_debug_labels
            .iter()
            .any(|label| label.starts_with("Debug Cargo"))
    );
}
