use collections::HashMap;
#[cfg(feature = "jvm-tools")]
use gpui::BorrowAppContext as _;
use gpui::{AppContext, Entity, Task, TestAppContext};
use itertools::Itertools;
use paths::tasks_file;
use pretty_assertions::assert_eq;
use serde_json::json;
use settings::SettingsLocation;
use std::path::Path;
#[cfg(feature = "jvm-tools")]
use std::path::PathBuf;
use std::sync::Arc;
use task::{RevealStrategy, RevealTarget, TaskTemplate};
#[cfg(feature = "jvm-tools")]
use task::{TaskContext, TaskVariables, VariableName};
#[cfg(feature = "jvm-tools")]
use util::path;
use util::rel_path::rel_path;

use project::task_store::{TaskSettingsLocation, TaskStore, plan_before_launch_dependencies};

#[cfg(feature = "jvm-tools")]
use fs::FakeFs;
#[cfg(feature = "jvm-tools")]
use project::Project;
use project::{WorktreeId, task_inventory::*};
use test_inventory::*;

mod test_inventory {
    use gpui::{AppContext as _, Entity, Task, TestAppContext};
    use itertools::Itertools;
    use task::TaskContext;
    use worktree::WorktreeId;

    use crate::Inventory;

    use super::TaskSourceKind;

    pub(super) fn task_template_names(
        inventory: &Entity<Inventory>,
        worktree: Option<WorktreeId>,
        cx: &mut TestAppContext,
    ) -> Task<Vec<String>> {
        let new_tasks = inventory.update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, worktree, cx)
        });
        cx.background_spawn(async move {
            new_tasks
                .await
                .into_iter()
                .map(|(_, task)| task.label)
                .sorted()
                .collect()
        })
    }

    pub(super) fn register_task_used(
        inventory: &Entity<Inventory>,
        task_name: &str,
        cx: &mut TestAppContext,
    ) -> Task<()> {
        let tasks = inventory.update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, None, cx)
        });

        let task_name = task_name.to_owned();
        let inventory = inventory.clone();
        cx.spawn(|mut cx| async move {
            let (task_source_kind, task) = tasks
                .await
                .into_iter()
                .find(|(_, task)| task.label == task_name)
                .unwrap_or_else(|| panic!("Failed to find task with name {task_name}"));

            let id_base = task_source_kind.to_id_base();
            inventory.update(&mut cx, |inventory, _| {
                inventory.task_scheduled(
                    task_source_kind.clone(),
                    task.resolve_task(&id_base, &TaskContext::default())
                        .unwrap_or_else(|| panic!("Failed to resolve task with name {task_name}")),
                )
            });
        })
    }

    pub(super) fn register_worktree_task_used(
        inventory: &Entity<Inventory>,
        worktree_id: WorktreeId,
        task_name: &str,
        cx: &mut TestAppContext,
    ) -> Task<()> {
        let tasks = inventory.update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(worktree_id), cx)
        });

        let inventory = inventory.clone();
        let task_name = task_name.to_owned();
        cx.spawn(|mut cx| async move {
            let (task_source_kind, task) = tasks
                .await
                .into_iter()
                .find(|(_, task)| task.label == task_name)
                .unwrap_or_else(|| panic!("Failed to find task with name {task_name}"));
            let id_base = task_source_kind.to_id_base();
            inventory.update(&mut cx, |inventory, _| {
                inventory.task_scheduled(
                    task_source_kind.clone(),
                    task.resolve_task(&id_base, &TaskContext::default())
                        .unwrap_or_else(|| panic!("Failed to resolve task with name {task_name}")),
                );
            });
        })
    }

    pub(super) async fn list_tasks(
        inventory: &Entity<Inventory>,
        worktree: Option<WorktreeId>,
        cx: &mut TestAppContext,
    ) -> Vec<(TaskSourceKind, String)> {
        let task_context = &TaskContext::default();
        inventory
            .update(cx, |inventory, cx| {
                inventory.list_tasks(None, None, worktree, cx)
            })
            .await
            .into_iter()
            .filter_map(|(source_kind, task)| {
                let id_base = source_kind.to_id_base();
                Some((source_kind, task.resolve_task(&id_base, task_context)?))
            })
            .map(|(source_kind, resolved_task)| (source_kind, resolved_task.resolved_label))
            .collect()
    }
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn groovy_build_tasks_require_detected_project_wrappers(cx: &mut TestAppContext) {
    use project::task_inventory::JvmBuildTool;

    crate::init_test(cx);
    let fs = FakeFs::new(cx.executor());
    let (maven_wrapper, gradle_wrapper, maven_command, gradle_command) = if cfg!(windows) {
        ("mvnw.cmd", "gradlew.bat", ".\\mvnw.cmd", ".\\gradlew.bat")
    } else {
        ("mvnw", "gradlew", "./mvnw", "./gradlew")
    };
    fs.insert_tree(
        path!("/jvm"),
        json!({
            "pom.xml": "<project/>",
            (maven_wrapper): "",
            "build.gradle.kts": "plugins { java }",
            (gradle_wrapper): ""
        }),
    )
    .await;
    fs.insert_tree(
        path!("/without-wrapper"),
        json!({ "pom.xml": "<project/>" }),
    )
    .await;
    fs.insert_tree(
        path!("/ignored-wrapper"),
        json!({
            ".gitignore": format!("{maven_wrapper}\n"),
            "pom.xml": "<project/>",
            (maven_wrapper): ""
        }),
    )
    .await;

    let project = Project::test(
        fs,
        [
            Path::new(path!("/jvm")),
            Path::new(path!("/without-wrapper")),
            Path::new(path!("/ignored-wrapper")),
        ],
        cx,
    )
    .await;
    let (inventory, worktrees) = project.read_with(cx, |project, cx| {
        let inventory = project
            .task_store()
            .read(cx)
            .task_inventory()
            .cloned()
            .expect("local project task inventory");
        let worktrees = project
            .worktree_store()
            .read(cx)
            .worktrees()
            .map(|worktree| {
                let worktree = worktree.read(cx);
                let expected = if worktree.root_name_str() == "jvm" {
                    (Some(true), Some(true))
                } else {
                    (Some(false), None)
                };
                assert_eq!(
                    (
                        JvmBuildTool::Maven.wrapper_detected(worktree),
                        JvmBuildTool::Gradle.wrapper_detected(worktree)
                    ),
                    expected
                );
                (worktree.root_name_str().to_string(), worktree.id())
            })
            .collect::<std::collections::HashMap<_, _>>();
        (inventory, worktrees)
    });

    let jvm_worktree = worktrees["jvm"];
    let tasks = inventory
        .update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(jvm_worktree), cx)
        })
        .await;
    let builds = tasks
        .into_iter()
        .filter(|(_, task)| task.label.starts_with("Build "))
        .collect::<Vec<_>>();
    assert_eq!(builds.len(), 2);
    assert!(builds.iter().any(|(_, task)| {
        task.label == "Build Maven project"
            && task.command == maven_command
            && task.args == ["package"]
            && task.cwd.as_deref() == Some("$ZED_WORKTREE_ROOT")
    }));
    assert!(builds.iter().any(|(_, task)| {
        task.label == "Build Gradle project"
            && task.command == gradle_command
            && task.args == ["build"]
            && task.cwd.as_deref() == Some("$ZED_WORKTREE_ROOT")
    }));
    let mut task_variables = TaskVariables::default();
    task_variables.insert(VariableName::WorktreeRoot, path!("/jvm").to_string());
    let task_context = TaskContext {
        cwd: Some(PathBuf::from(path!("/jvm"))),
        task_variables,
        ..TaskContext::default()
    };
    for (source, template) in &builds {
        let resolved = template
            .resolve_task(&source.to_id_base(), &task_context)
            .expect("detected wrapper task resolves in its host worktree");
        assert_eq!(resolved.resolved.cwd, task_context.cwd);
    }

    let unwrapped_worktree = worktrees["without-wrapper"];
    let unwrapped_tasks = inventory
        .update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(unwrapped_worktree), cx)
        })
        .await;
    assert!(
        !unwrapped_tasks
            .iter()
            .any(|(_, task)| task.label.starts_with("Build "))
    );

    let ignored_worktree = worktrees["ignored-wrapper"];
    project.read_with(cx, |project, cx| {
        let worktree = project
            .worktrees(cx)
            .find(|worktree| worktree.read(cx).id() == ignored_worktree)
            .expect("ignored-wrapper worktree");
        assert!(
            worktree
                .read(cx)
                .snapshot()
                .entry_for_path(rel_path("mvnw"))
                .expect("ignored Maven wrapper")
                .is_ignored
        );
    });
    let ignored_tasks = inventory
        .update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(ignored_worktree), cx)
        })
        .await;
    assert!(
        !ignored_tasks
            .iter()
            .any(|(_, task)| task.label == "Build Maven project")
    );

    cx.update(|cx| {
        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
            store.update_user_settings(cx, |settings| {
                settings.project.worktree.private_files =
                    Some(settings::ExtendingVec(vec!["**/mvnw".into()]));
            });
        });
    });
    cx.run_until_parked();
    project.read_with(cx, |project, cx| {
        let worktree = project
            .worktrees(cx)
            .find(|worktree| worktree.read(cx).id() == jvm_worktree)
            .expect("JVM worktree");
        assert!(
            worktree
                .read(cx)
                .snapshot()
                .entry_for_path(rel_path("mvnw"))
                .expect("Maven wrapper")
                .is_private
        );
    });
    let tasks = inventory
        .update(cx, |inventory, cx| {
            inventory.list_tasks(None, None, Some(jvm_worktree), cx)
        })
        .await;
    assert!(
        !tasks
            .iter()
            .any(|(_, task)| task.label == "Build Maven project")
    );
    assert!(
        tasks
            .iter()
            .any(|(_, task)| task.label == "Build Gradle project")
    );
}

#[gpui::test]
async fn test_task_list_sorting(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    let initial_tasks = resolved_task_names(&inventory, None, cx).await;
    assert!(
        initial_tasks.is_empty(),
        "No tasks expected for empty inventory, but got {initial_tasks:?}"
    );
    let initial_tasks = task_template_names(&inventory, None, cx).await;
    assert!(
        initial_tasks.is_empty(),
        "No tasks expected for empty inventory, but got {initial_tasks:?}"
    );
    cx.run_until_parked();
    let expected_initial_state = [
        "1_a_task".to_string(),
        "1_task".to_string(),
        "2_task".to_string(),
        "3_task".to_string(),
    ];

    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(&mock_tasks_from_names(
                    expected_initial_state.iter().map(|name| name.as_str()),
                )),
            )
            .unwrap();
    });
    assert_eq!(
        task_template_names(&inventory, None, cx).await,
        &expected_initial_state,
    );
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        &expected_initial_state,
        "Tasks with equal amount of usages should be sorted alphanumerically"
    );

    register_task_used(&inventory, "2_task", cx).await;
    assert_eq!(
        task_template_names(&inventory, None, cx).await,
        &expected_initial_state,
    );
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        vec![
            "2_task".to_string(),
            "1_a_task".to_string(),
            "1_task".to_string(),
            "3_task".to_string()
        ],
    );

    register_task_used(&inventory, "1_task", cx).await;
    register_task_used(&inventory, "1_task", cx).await;
    register_task_used(&inventory, "1_task", cx).await;
    register_task_used(&inventory, "3_task", cx).await;
    assert_eq!(
        task_template_names(&inventory, None, cx).await,
        &expected_initial_state,
    );
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        vec![
            "3_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "1_a_task".to_string(),
        ],
        "Most recently used task should be at the top"
    );

    let worktree_id = WorktreeId::from_usize(0);
    let local_worktree_location = SettingsLocation {
        worktree_id,
        path: rel_path("foo"),
    };
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(local_worktree_location),
                Some(&mock_tasks_from_names(["worktree_task_1"])),
            )
            .unwrap();
    });
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        vec![
            "3_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "1_a_task".to_string(),
        ],
        "Most recently used task should be at the top"
    );
    assert_eq!(
        resolved_task_names(&inventory, Some(worktree_id), cx).await,
        vec![
            "3_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "worktree_task_1".to_string(),
            "1_a_task".to_string(),
        ],
    );
    register_worktree_task_used(&inventory, worktree_id, "worktree_task_1", cx).await;
    assert_eq!(
        resolved_task_names(&inventory, Some(worktree_id), cx).await,
        vec![
            "worktree_task_1".to_string(),
            "3_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "1_a_task".to_string(),
        ],
        "Most recently used worktree task should be at the top"
    );

    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(&mock_tasks_from_names(
                    ["10_hello", "11_hello"]
                        .into_iter()
                        .chain(expected_initial_state.iter().map(|name| name.as_str())),
                )),
            )
            .unwrap();
    });
    cx.run_until_parked();
    let expected_updated_state = [
        "10_hello".to_string(),
        "11_hello".to_string(),
        "1_a_task".to_string(),
        "1_task".to_string(),
        "2_task".to_string(),
        "3_task".to_string(),
    ];
    assert_eq!(
        task_template_names(&inventory, None, cx).await,
        &expected_updated_state,
    );
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        vec![
            "worktree_task_1".to_string(),
            "1_a_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "3_task".to_string(),
            "10_hello".to_string(),
            "11_hello".to_string(),
        ],
        "After global tasks update, worktree task usage is not erased and it's the first still; global task is back to regular order as its file was updated"
    );

    register_task_used(&inventory, "11_hello", cx).await;
    assert_eq!(
        task_template_names(&inventory, None, cx).await,
        &expected_updated_state,
    );
    assert_eq!(
        resolved_task_names(&inventory, None, cx).await,
        vec![
            "11_hello".to_string(),
            "worktree_task_1".to_string(),
            "1_a_task".to_string(),
            "1_task".to_string(),
            "2_task".to_string(),
            "3_task".to_string(),
            "10_hello".to_string(),
        ],
    );
}

#[gpui::test]
#[gpui::test]
async fn test_saving_named_run_configurations(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(&mock_tasks_from_names(["other_task", "provider-build"])),
            )
            .unwrap();
    });

    let configuration = TaskTemplate {
        label: "provider-build".into(),
        command: "mvn".into(),
        args: vec![
            "--offline".into(),
            "-Dmaven.repo.local=/cache".into(),
            "-Djvm.options=-Xmx2g".into(),
            "test".into(),
        ],
        env: HashMap::from_iter([("MAVEN_OPTS".into(), "-Xmx2g".into())]),
        cwd: Some("target/".into()),
        allow_concurrent_runs: false,
        reveal: RevealStrategy::Never,
        reveal_target: RevealTarget::Center,
        show_summary: true,
        show_command: false,
        ..Default::default()
    };
    let saved_json = inventory.update(cx, |inventory, _| {
        inventory
            .save_named_run_configuration(
                TaskSettingsLocation::Global(tasks_file()),
                configuration.clone(),
            )
            .unwrap()
    });
    let saved = serde_json::from_str::<Vec<serde_json::Value>>(&saved_json).unwrap();
    let labels: Vec<&str> = saved
        .iter()
        .map(|task| task["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["other_task", "provider-build"]);
    let entry = saved
        .iter()
        .find(|task| task["label"] == "provider-build")
        .unwrap();
    assert_eq!(entry["command"], "mvn");
    assert_eq!(entry["cwd"], "target/");
    assert_eq!(entry["allow_concurrent_runs"], false);
    assert_eq!(entry["env"]["MAVEN_OPTS"], "-Xmx2g");
    assert_eq!(entry["show_summary"], true);
    assert_eq!(entry["show_command"], false);
    assert_eq!(entry["reveal"], "never");
    assert_eq!(entry["reveal_target"], "center");

    // The existing file-based owner admitted the merged content; the saved
    // configuration resolves to the same template without any field loss.
    let admitted = inventory
        .update(cx, |inventory, cx| {
            inventory.task_template_by_label(None, None, "provider-build", cx)
        })
        .await;
    assert_eq!(admitted.as_ref(), Some(&configuration));

    // Saving again replaces the same-label entry in place.
    let updated = TaskTemplate {
        args: vec!["--offline".into(), "test".into()],
        ..configuration.clone()
    };
    let resaved_json = inventory.update(cx, |inventory, _| {
        inventory
            .save_named_run_configuration(
                TaskSettingsLocation::Global(tasks_file()),
                updated.clone(),
            )
            .unwrap()
    });
    let resaved = serde_json::from_str::<Vec<serde_json::Value>>(&resaved_json).unwrap();
    assert_eq!(
        resaved
            .iter()
            .filter(|task| task["label"] == "provider-build")
            .count(),
        1
    );
    let admitted = inventory
        .update(cx, |inventory, cx| {
            inventory.task_template_by_label(None, None, "provider-build", cx)
        })
        .await;
    assert_eq!(admitted.as_ref(), Some(&updated));

    // A configuration that uses unknown variables is rejected before it can
    // reach the shared owner.
    let invalid = TaskTemplate {
        label: "broken".into(),
        command: "echo".into(),
        args: vec!["${ZED_NOT_A_REAL_VARIABLE}".into()],
        ..Default::default()
    };
    let error = inventory
        .update(cx, |inventory, _| {
            inventory
                .save_named_run_configuration(TaskSettingsLocation::Global(tasks_file()), invalid)
        })
        .unwrap_err()
        .to_string();
    assert!(error.contains("uses unknown variables"), "{error}");

    // Worktree-scope configurations are retained under their worktree only.
    let worktree_id = WorktreeId::from_usize(0);
    let worktree_configuration = TaskTemplate {
        label: "provider-worktree-build".into(),
        command: "gradle".into(),
        args: vec!["--offline".into(), "test".into()],
        ..Default::default()
    };
    inventory.update(cx, |inventory, _| {
        inventory
            .save_named_run_configuration(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id,
                    path: rel_path(".zed"),
                }),
                worktree_configuration.clone(),
            )
            .unwrap();
    });
    let admitted = inventory
        .update(cx, |inventory, cx| {
            inventory.task_template_by_label(None, Some(worktree_id), "provider-worktree-build", cx)
        })
        .await;
    assert_eq!(admitted.as_ref(), Some(&worktree_configuration));
    let admitted = inventory
        .update(cx, |inventory, cx| {
            inventory.task_template_by_label(None, None, "provider-worktree-build", cx)
        })
        .await;
    assert!(
        admitted.is_none(),
        "a worktree-saved configuration is not a user configuration"
    );
}

#[gpui::test]
async fn test_planning_before_launch_dependencies(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(&mock_tasks_from_names([
                    "sync-first",
                    "build-second",
                    "standalone",
                ])),
            )
            .unwrap();
    });
    let handle = inventory.clone();
    let plan = |launching: &str,
                dependencies: &[&str],
                cx: &mut TestAppContext|
     -> anyhow::Result<Task<anyhow::Result<Vec<task::ResolvedTask>>>> {
        let dependencies: Vec<String> = dependencies.iter().map(|l| l.to_string()).collect();
        cx.update(|cx| {
            plan_before_launch_dependencies(
                &handle,
                None,
                None,
                launching,
                &dependencies,
                &task::TaskContext::default(),
                cx,
            )
        })
    };

    // Ordered dependencies resolve in the declared order through the saved
    // task inventory with the launching task's context.
    let planned = plan("provider-build", &["sync-first", "build-second"], cx)
        .unwrap()
        .await
        .unwrap();
    assert_eq!(planned.len(), 2);
    assert_eq!(planned[0].resolved_label, "sync-first");
    assert_eq!(planned[1].resolved_label, "build-second");
    assert_eq!(planned[0].original_task().label, "sync-first");

    // An empty dependency list plans nothing.
    assert!(
        plan("provider-build", &[], cx)
            .unwrap()
            .await
            .unwrap()
            .is_empty()
    );

    // A missing saved task is an explicit error.
    let error = plan("provider-build", &["missing-task"], cx)
        .unwrap()
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("is not a saved task"), "{error}");

    // Duplicates and self-references are rejected before scheduling.
    let error = plan("provider-build", &["sync-first", "sync-first"], cx)
        .unwrap_err()
        .to_string();
    assert!(error.contains("listed more than once"), "{error}");
    let error = plan("provider-build", &["provider-build"], cx)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("cannot reference its own configuration"),
        "{error}"
    );

    // The dependency count is bounded.
    let over_limit: Vec<String> = (0..plan_before_launch_dependencies_len_limit() + 1)
        .map(|index| format!("dep-{index}"))
        .collect();
    let over_limit: Vec<&str> = over_limit.iter().map(String::as_str).collect();
    let error = plan("provider-build", &over_limit, cx)
        .unwrap_err()
        .to_string();
    assert!(error.contains("-dependency limit"), "{error}");
}

fn plan_before_launch_dependencies_len_limit() -> usize {
    project::task_store::MAX_BEFORE_LAUNCH_DEPENDENCIES
}

#[cfg(feature = "jvm-tools")]
#[gpui::test]
async fn test_planning_jvm_trigger_sequences(cx: &mut TestAppContext) {
    use project::task_store::{
        JvmTaskTrigger, MAX_TRIGGER_SEQUENCE_DEPENDENCIES, plan_trigger_sequence,
    };

    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    let initial_tasks = serde_json::to_string(&serde_json::Value::Array(
        vec![
            ("sync-setup", vec!["before-sync"]),
            ("build-check", vec!["after-build"]),
            ("both-phases", vec!["before-sync", "after-run"]),
            ("untagged", Vec::<&str>::new()),
            ("unrelated", vec!["coverage"]),
        ]
        .into_iter()
        .map(|(label, tags)| {
            json!({
                "label": label,
                "command": "echo",
                "args": vec![label],
                "tags": tags,
            })
        })
        .collect::<Vec<_>>(),
    ))
    .unwrap();
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(initial_tasks.as_str()),
            )
            .unwrap();
    });

    // Only configurations tagged for the trigger join its ordered sequence.
    let planned =
        cx.update(|cx| plan_trigger_sequence(&inventory, None, JvmTaskTrigger::BeforeSync, cx));
    let sequence = planned.await.unwrap();
    assert_eq!(sequence.len(), 2);
    assert_eq!(sequence[0].1.label, "sync-setup");
    assert_eq!(sequence[1].1.label, "both-phases");

    // Unrelated tags never authorize a trigger sequence.
    let planned =
        cx.update(|cx| plan_trigger_sequence(&inventory, None, JvmTaskTrigger::AfterDebug, cx));
    assert!(planned.await.unwrap().is_empty());

    // The same configuration can serve two different triggers through its
    // separate tags.
    let planned =
        cx.update(|cx| plan_trigger_sequence(&inventory, None, JvmTaskTrigger::AfterRun, cx));
    let sequence = planned.await.unwrap();
    assert_eq!(sequence.len(), 1);
    assert_eq!(sequence[0].1.label, "both-phases");

    // Duplicate labels and over-limit sequences are rejected explicitly.
    let duplicate_tasks = serde_json::to_string(&serde_json::Value::Array(
        vec![
            ("repeat", vec!["before-rebuild"]),
            ("repeat", vec!["before-rebuild"]),
        ]
        .into_iter()
        .map(|(label, tags)| {
            json!({
                "label": label,
                "command": "echo",
                "args": vec![label],
                "tags": tags,
            })
        })
        .collect::<Vec<_>>(),
    ))
    .unwrap();
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(Path::new("/duplicate/tasks.json")),
                Some(duplicate_tasks.as_str()),
            )
            .unwrap();
    });
    let planned =
        cx.update(|cx| plan_trigger_sequence(&inventory, None, JvmTaskTrigger::BeforeRebuild, cx));
    let error = planned.await.unwrap_err().to_string();
    assert!(error.contains("more than once"), "{error}");

    let over_limit: Vec<(String, String)> = (0..MAX_TRIGGER_SEQUENCE_DEPENDENCIES + 1)
        .map(|index| {
            let label = format!("over-{index}");
            let tasks = serde_json::to_string(&serde_json::Value::Array(vec![json!({
                "label": label,
                "command": "echo",
                "args": vec![label.clone()],
                "tags": vec!["after-sync"],
            })]))
            .unwrap();
            (label, tasks)
        })
        .collect();
    for (label, tasks) in &over_limit {
        let path = format!("/limit/{label}.json");
        inventory.update(cx, |inventory, _| {
            inventory
                .update_file_based_tasks(
                    TaskSettingsLocation::Global(Path::new(path.as_str())),
                    Some(tasks.as_str()),
                )
                .unwrap();
        });
    }
    let planned =
        cx.update(|cx| plan_trigger_sequence(&inventory, None, JvmTaskTrigger::AfterSync, cx));
    let error = planned.await.unwrap_err().to_string();
    assert!(error.contains("limit"), "{error}");
}

#[gpui::test]
async fn test_reloading_debug_scenarios(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_scenarios(
                TaskSettingsLocation::Global(Path::new("")),
                Some(
                    r#"
                        [{
                            "label": "test scenario",
                            "adapter": "CodeLLDB",
                            "request": "launch",
                            "program": "wowzer",
                        }]
                        "#,
                ),
            )
            .unwrap();
    });

    let (_, scenario) = inventory
        .update(cx, |this, cx| {
            this.list_debug_scenarios(&TaskContexts::default(), vec![], vec![], false, cx)
        })
        .await
        .1
        .first()
        .unwrap()
        .clone();

    inventory.update(cx, |this, _| {
        this.scenario_scheduled(scenario.clone(), Default::default(), None, None);
    });

    assert_eq!(
        inventory
            .update(cx, |this, cx| {
                this.list_debug_scenarios(&Default::default(), vec![], vec![], false, cx)
            })
            .await
            .0
            .first()
            .unwrap()
            .clone()
            .0,
        scenario
    );

    inventory.update(cx, |this, _| {
        this.update_file_based_scenarios(
            TaskSettingsLocation::Global(Path::new("")),
            Some(
                r#"
                        [{
                            "label": "test scenario",
                            "adapter": "Delve",
                            "request": "launch",
                            "program": "wowzer",
                        }]
                        "#,
            ),
        )
        .unwrap();
    });

    assert_eq!(
        inventory
            .update(cx, |this, cx| {
                this.list_debug_scenarios(&Default::default(), vec![], vec![], false, cx)
            })
            .await
            .0
            .first()
            .unwrap()
            .0
            .adapter,
        "Delve",
    );

    inventory.update(cx, |this, _| {
        this.update_file_based_scenarios(
            TaskSettingsLocation::Global(Path::new("")),
            Some(
                r#"
                        [{
                            "label": "testing scenario",
                            "adapter": "Delve",
                            "request": "launch",
                            "program": "wowzer",
                        }]
                        "#,
            ),
        )
        .unwrap();
    });

    assert!(
        inventory
            .update(cx, |this, cx| {
                this.list_debug_scenarios(&TaskContexts::default(), vec![], vec![], false, cx)
            })
            .await
            .0
            .is_empty(),
    );
}

#[gpui::test]
async fn test_inventory_static_task_filters(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    let common_name = "common_task_name";
    let worktree_1 = WorktreeId::from_usize(1);
    let worktree_2 = WorktreeId::from_usize(2);

    cx.run_until_parked();
    let worktree_independent_tasks = vec![
        (
            TaskSourceKind::AbsPath {
                id_base: "global tasks.json".into(),
                abs_path: paths::tasks_file().clone(),
            },
            common_name.to_string(),
        ),
        (
            TaskSourceKind::AbsPath {
                id_base: "global tasks.json".into(),
                abs_path: paths::tasks_file().clone(),
            },
            "static_source_1".to_string(),
        ),
        (
            TaskSourceKind::AbsPath {
                id_base: "global tasks.json".into(),
                abs_path: paths::tasks_file().clone(),
            },
            "static_source_2".to_string(),
        ),
    ];
    let worktree_1_tasks = [
        (
            TaskSourceKind::Worktree {
                id: worktree_1,
                directory_in_worktree: rel_path(".zed").into(),
                id_base: "local worktree tasks from directory \".zed\"".into(),
            },
            common_name.to_string(),
        ),
        (
            TaskSourceKind::Worktree {
                id: worktree_1,
                directory_in_worktree: rel_path(".zed").into(),
                id_base: "local worktree tasks from directory \".zed\"".into(),
            },
            "worktree_1".to_string(),
        ),
    ];
    let worktree_2_tasks = [
        (
            TaskSourceKind::Worktree {
                id: worktree_2,
                directory_in_worktree: rel_path(".zed").into(),
                id_base: "local worktree tasks from directory \".zed\"".into(),
            },
            common_name.to_string(),
        ),
        (
            TaskSourceKind::Worktree {
                id: worktree_2,
                directory_in_worktree: rel_path(".zed").into(),
                id_base: "local worktree tasks from directory \".zed\"".into(),
            },
            "worktree_2".to_string(),
        ),
    ];

    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Global(tasks_file()),
                Some(&mock_tasks_from_names(
                    worktree_independent_tasks
                        .iter()
                        .map(|(_, name)| name.as_str()),
                )),
            )
            .unwrap();
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id: worktree_1,
                    path: rel_path(".zed"),
                }),
                Some(&mock_tasks_from_names(
                    worktree_1_tasks.iter().map(|(_, name)| name.as_str()),
                )),
            )
            .unwrap();
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id: worktree_2,
                    path: rel_path(".zed"),
                }),
                Some(&mock_tasks_from_names(
                    worktree_2_tasks.iter().map(|(_, name)| name.as_str()),
                )),
            )
            .unwrap();
    });

    assert_eq!(
        list_tasks_sorted_by_last_used(&inventory, None, cx).await,
        worktree_independent_tasks,
        "Without a worktree, only worktree-independent tasks should be listed"
    );
    assert_eq!(
        list_tasks_sorted_by_last_used(&inventory, Some(worktree_1), cx).await,
        worktree_1_tasks
            .iter()
            .chain(worktree_independent_tasks.iter())
            .cloned()
            .sorted_by_key(|(kind, label)| (task_source_kind_preference(kind), label.clone()))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        list_tasks_sorted_by_last_used(&inventory, Some(worktree_2), cx).await,
        worktree_2_tasks
            .iter()
            .chain(worktree_independent_tasks.iter())
            .cloned()
            .sorted_by_key(|(kind, label)| (task_source_kind_preference(kind), label.clone()))
            .collect::<Vec<_>>(),
    );

    assert_eq!(
        list_tasks(&inventory, None, cx).await,
        worktree_independent_tasks,
        "Without a worktree, only worktree-independent tasks should be listed"
    );
    assert_eq!(
        list_tasks(&inventory, Some(worktree_1), cx).await,
        worktree_1_tasks
            .iter()
            .chain(worktree_independent_tasks.iter())
            .cloned()
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        list_tasks(&inventory, Some(worktree_2), cx).await,
        worktree_2_tasks
            .iter()
            .chain(worktree_independent_tasks.iter())
            .cloned()
            .collect::<Vec<_>>(),
    );
}

#[gpui::test]
async fn test_zed_tasks_take_precedence_over_vscode(cx: &mut TestAppContext) {
    init_test(cx);
    let inventory = cx.update(|cx| Inventory::new(cx));
    let worktree_id = WorktreeId::from_usize(0);

    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id,
                    path: rel_path(".vscode"),
                }),
                Some(&mock_tasks_from_names(["vscode_task"])),
            )
            .unwrap();
    });
    assert_eq!(
        task_template_names(&inventory, Some(worktree_id), cx).await,
        vec!["vscode_task"],
        "With only .vscode tasks, they should appear"
    );

    inventory.update(cx, |inventory, _| {
        inventory
            .update_file_based_tasks(
                TaskSettingsLocation::Worktree(SettingsLocation {
                    worktree_id,
                    path: rel_path(".zed"),
                }),
                Some(&mock_tasks_from_names(["zed_task"])),
            )
            .unwrap();
    });
    assert_eq!(
        task_template_names(&inventory, Some(worktree_id), cx).await,
        vec!["zed_task"],
        "With both .zed and .vscode tasks, only .zed tasks should appear"
    );

    register_worktree_task_used(&inventory, worktree_id, "zed_task", cx).await;
    let resolved = resolved_task_names(&inventory, Some(worktree_id), cx).await;
    assert!(
        !resolved.iter().any(|name| name == "vscode_task"),
        "Previously used .vscode tasks should not appear when .zed tasks exist, got: {resolved:?}"
    );
}

fn init_test(_cx: &mut TestAppContext) {
    zlog::init_test();
    TaskStore::init(None);
}

fn resolved_task_names(
    inventory: &Entity<Inventory>,
    worktree: Option<WorktreeId>,
    cx: &mut TestAppContext,
) -> Task<Vec<String>> {
    let tasks = inventory.update(cx, |inventory, cx| {
        let mut task_contexts = TaskContexts::default();
        task_contexts.active_worktree_context =
            worktree.map(|worktree| (worktree, Default::default()));

        inventory.used_and_current_resolved_tasks(Arc::new(task_contexts), cx)
    });

    cx.background_spawn(async move {
        let (used, current) = tasks.await;
        used.into_iter()
            .chain(current)
            .map(|(_, task)| task.original_task().label.clone())
            .collect()
    })
}

fn mock_tasks_from_names<'a>(task_names: impl IntoIterator<Item = &'a str> + 'a) -> String {
    serde_json::to_string(&serde_json::Value::Array(
        task_names
            .into_iter()
            .map(|task_name| {
                json!({
                    "label": task_name,
                    "command": "echo",
                    "args": vec![task_name],
                })
            })
            .collect::<Vec<_>>(),
    ))
    .unwrap()
}

async fn list_tasks_sorted_by_last_used(
    inventory: &Entity<Inventory>,
    worktree: Option<WorktreeId>,
    cx: &mut TestAppContext,
) -> Vec<(TaskSourceKind, String)> {
    let (used, current) = inventory
        .update(cx, |inventory, cx| {
            let mut task_contexts = TaskContexts::default();
            task_contexts.active_worktree_context =
                worktree.map(|worktree| (worktree, Default::default()));

            inventory.used_and_current_resolved_tasks(Arc::new(task_contexts), cx)
        })
        .await;
    let mut all = used;
    all.extend(current);
    all.into_iter()
        .map(|(source_kind, task)| (source_kind, task.resolved_label))
        .sorted_by_key(|(kind, label)| (task_source_kind_preference(kind), label.clone()))
        .collect()
}
