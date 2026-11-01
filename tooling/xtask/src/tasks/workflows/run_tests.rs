use gh_workflow::{
    Container, Event, Expression, Job, Port, PullRequest, Push, Run, Step, Strategy, Use, Workflow,
    WorkflowDispatch,
};
use indexmap::IndexMap;
use indoc::formatdoc;
use serde_json::json;

use crate::product_manifest::ProductManifest;

use crate::tasks::workflows::{
    steps::{
        CommonJobConditions, CommonPermissionSets, repository_owner_guard_expression, use_clang,
    },
    vars::{self, PathCondition},
};

use super::{
    runners::{self, Platform},
    steps::{self, FluentBuilder, NamedJob, named, release_job},
};

pub(crate) fn run_tests() -> Workflow {
    let shared_clippy = shared_clippy();
    let copper_tests = copper_tests();
    let graphics_feature_validation = graphics_feature_validation();
    let project_benchmarks = project_benchmarks();
    let rust_tools_validation = rust_tools_validation();
    let release_automation_validation = release_automation_validation();
    let windows_updater_tests = windows_updater_tests();
    let windows_jvm_validation = windows_jvm_validation();
    let multiplayer_tools_standard = multiplayer_tools_profile("standard");
    let multiplayer_tools_multiplayer = multiplayer_tools_profile("multiplayer");
    let validation = shared_validation(&[
        &shared_clippy,
        &copper_tests,
        &graphics_feature_validation,
        &project_benchmarks,
        &rust_tools_validation,
        &release_automation_validation,
        &windows_updater_tests,
        &windows_jvm_validation,
        &multiplayer_tools_standard,
        &multiplayer_tools_multiplayer,
    ]);
    let sim_backend_validation = sim_backend_validation();
    let product_smoke = product_smoke(&validation);

    named::workflow()
        .with_minimal_permissions()
        .on(Event::default()
            .push(Push::default().add_branch("main"))
            .pull_request(PullRequest::default().add_branch("**"))
            .workflow_dispatch(WorkflowDispatch::default()))
        .concurrency(vars::one_workflow_per_non_main_branch())
        .add_env(("CARGO_TERM_COLOR", "always"))
        .add_env(("RUST_BACKTRACE", 1))
        .add_env(("CARGO_INCREMENTAL", 0))
        .add_job(shared_clippy.name, shared_clippy.job)
        .add_job(copper_tests.name, copper_tests.job)
        .add_job(
            graphics_feature_validation.name,
            graphics_feature_validation.job,
        )
        .add_job(project_benchmarks.name, project_benchmarks.job)
        .add_job(rust_tools_validation.name, rust_tools_validation.job)
        .add_job(
            release_automation_validation.name,
            release_automation_validation.job,
        )
        .add_job(
            multiplayer_tools_standard.name,
            multiplayer_tools_standard.job,
        )
        .add_job(
            multiplayer_tools_multiplayer.name,
            multiplayer_tools_multiplayer.job,
        )
        .add_job(validation.name, validation.job)
        .add_job(windows_updater_tests.name, windows_updater_tests.job)
        .add_job(windows_jvm_validation.name, windows_jvm_validation.job)
        .add_job(sim_backend_validation.name, sim_backend_validation.job)
        .add_job(product_smoke.name, product_smoke.job)
}

fn multiplayer_tools_profile(profile: &str) -> NamedJob {
    NamedJob {
        name: format!("multiplayer_tools_{profile}"),
        job: Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(120u32)
            .add_env(("ZED_PRODUCT_ID", RUST_PRODUCT_ID))
            .add_env(("CARGO_PROFILE_TEST_DEBUG", 0))
            .add_env(("CARGO_PROFILE_DEV_DEBUG", 0))
            .add_env(("CARGO_PROFILE_RELEASE_DEBUG", 0))
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .add_step(steps::setup_linux())
            .add_step(setup_inventory_python())
            .add_step(named::bash(format!(
                "./script/check-multiplayer-tools --full --profile {profile}"
            )))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    }
}

fn graphics_feature_validation() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(120u32)
            .add_env(("ZED_PRODUCT_ID", RUST_PRODUCT_ID))
            .add_env(("CARGO_PROFILE_TEST_DEBUG", 0))
            .add_env(("CARGO_PROFILE_DEV_DEBUG", 0))
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .map(steps::install_linux_dependencies)
            .add_step(named::bash(indoc::indoc! {r#"
                cargo test --locked -p graphics_core -p graphics_ui --lib
                cargo test --locked -p graphics_core --test native_worker
                cargo test --locked -p gpui --lib tab_stop::tests::
                cargo test --locked -p zed --no-default-features --features agentic-tools,sim --bin zed graphics_surface_respects_sim_feature
                cargo test --locked -p zed --no-default-features --features agentic-tools,sim --bin zed native_sim_
                ZED_PRODUCT_ID=jvm ZED_PRODUCT_DEV_PREVIEW=1 ZED_RELEASE_CHANNEL=dev cargo test --locked -p zed --no-default-features --features agentic-tools,jvm-tools --bin zed graphics_surface_respects_sim_feature
            "#}))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn shared_clippy() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(60u32)
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .add_step(steps::setup_linux())
            .add_step(steps::cargo_fmt())
            .add_step(named::bash("./script/clippy"))
            .add_step(setup_inventory_python())
            .add_step(named::bash(indoc::indoc! {r#"
                if git cat-file -e HEAD:projects/buzz/Cargo.toml 2>/dev/null; then
                  ./script/check-collaborative-workspace-inventory --catalog all
                else
                  echo '::notice::Collaborative workspace source is absent; full inventory audit deferred'
                fi
            "#}))
            .add_step(named::bash(
                "python3 script/test-collaborative-workspace-inventory.py",
            ))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn setup_inventory_python() -> Step<Use> {
    named::uses(
        "actions",
        "setup-python",
        "a26af69be951a213d495a4c3e4e4022e16d87065", // v5.6.0
    )
    .add_with(("python-version", "3.12"))
}

const RUST_PRODUCT_ID: &str = "rust";
const RUST_PRODUCT_TEST_UPDATE_BASE_URL: &str = "http://test.example";

const RUST_PRODUCT_TEST_PACKAGES: &[&str] = &[
    "agent",
    "agent_ui",
    "auto_update",
    "call",
    "cargo_ui",
    "channel",
    "client",
    "collab_ui",
    "editor",
    "extension_host",
    "git",
    "git_ui",
    "language_model",
    "language_models",
    "language_tools",
    "languages",
    "project",
    "remote_server",
    "settings",
    "settings_content",
    "task",
    "tasks_ui",
    "terminal",
    "terminal_view",
    "worktree",
    "workspace",
    "zed",
];

fn rust_product_nextest_command(measure_foreground: bool) -> String {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let product = manifest
        .product(RUST_PRODUCT_ID)
        .expect("Rust product must exist");
    let features = product
        .cargo_features
        .iter()
        .map(|feature| format!("zed/{feature}"))
        .chain(
            product
                .remote_server_features
                .iter()
                .map(|feature| format!("remote_server/{feature}")),
        )
        .collect::<Vec<_>>()
        .join(",");
    let packages = RUST_PRODUCT_TEST_PACKAGES
        .iter()
        .map(|package| format!("-p {package}"))
        .collect::<Vec<_>>()
        .join(" ");

    let selection = if measure_foreground {
        "-E '(test(=cargo_panel::tests::cargo_dashboard_foreground_budget) or test(=cargo_panel::tests::cargo_dashboard_production_foreground_budget) or test(=test_explorer::tests::structured_execution_foreground_budget) or test(=profile_view::tests::profile_view_deep_selection_expands_ancestors_in_one_batch))' --test-threads 1 --success-output immediate"
    } else {
        "-E 'not (test(=cargo_panel::tests::cargo_dashboard_foreground_budget) or test(=cargo_panel::tests::cargo_dashboard_production_foreground_budget) or test(=test_explorer::tests::structured_execution_foreground_budget) or test(=profile_view::tests::profile_view_deep_selection_expands_ancestors_in_one_batch))'"
    };
    let no_tests = if measure_foreground { "fail" } else { "warn" };
    format!(
        "cargo nextest run --no-fail-fast --no-tests={no_tests} --no-default-features --features {features} {packages} {selection}"
    )
}

fn copper_tests() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(90u32)
            .add_env(("CARGO_PROFILE_TEST_DEBUG", 0))
            .add_env(("SCCACHE_GHA_ENABLED", "on"))
            .add_env(("SCCACHE_IDLE_TIMEOUT", "0"))
            .add_env(("RUSTC_WRAPPER", "sccache"))
            .add_env(("ZED_PRODUCT_ID", RUST_PRODUCT_ID))
            .add_env((
                "ZED_PRODUCT_UPDATE_BASE_URL",
                RUST_PRODUCT_TEST_UPDATE_BASE_URL,
            ))
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .map(steps::install_linux_dependencies)
            .add_step(steps::setup_release_sccache())
            .add_step(
                Step::new("Configure Copper compiler cache")
                    .run(indoc::indoc! {r#"
                        TOOLCHAIN_HASH="$(git hash-object rust-toolchain.toml)"
                        echo "SCCACHE_GHA_VERSION=copper-tests-v1-${RUNNER_OS}-${RUNNER_ARCH}-${TOOLCHAIN_HASH}" >> "$GITHUB_ENV"
                    "#})
                    .shell(steps::BASH_SHELL),
            )
            .add_step(steps::setup_node())
            .add_step(steps::cargo_install_nextest())
            .add_step(named::bash(rust_product_nextest_command(false)))
            .add_step(
                Step::new("Measure Copper foreground budgets")
                    .run(rust_product_nextest_command(true)),
            )
            .add_step(
                steps::finalize_release_sccache(Platform::Linux)
                    .if_condition(Expression::new("always()")),
            )
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn project_benchmarks() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(75u32)
            .strategy(Strategy::default().fail_fast(false).matrix(json!({
                "include": [
                    { "benchmark": "cargo_workspace" },
                    { "benchmark": "structured_execution" },
                ],
            })))
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .add_step(steps::setup_linux())
            .add_step(
                named::bash("cargo bench -p project --features cargo-workspace --bench cargo_workspace -- --noplot")
                    .if_condition(Expression::new("matrix.benchmark == 'cargo_workspace'")),
            )
            .add_step(
                named::bash("cargo bench -p project --features structured-execution --bench structured_execution -- --noplot")
                    .if_condition(Expression::new("matrix.benchmark == 'structured_execution'")),
            )
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn rust_tools_validation() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(45u32)
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::free_linux_disk_space())
            .add_step(named::bash(
                "python3 script/test-rust-tools-environments-failures.py",
            ))
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .add_step(steps::setup_linux())
            .add_step(named::bash("cargo fetch --locked"))
            .add_step(named::bash(
                "cargo test --locked --offline -p native_task_helper --lib",
            ))
            .add_step(named::bash(
                "./script/test-rust-tools-environments --matrix --offline",
            ))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn release_automation_validation() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(30u32)
            .add_step(steps::checkout_repo())
            .add_step(setup_inventory_python())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .add_step(named::bash(indoc::indoc! {r#"
                python3 -m unittest discover -s script -p 'test_refresh_website.py'
                python3 -m unittest discover -s script -p 'test_product_release.py'
                python3 script/test-production-signing
                cargo test -p xtask
                cargo xtask workflows
                git diff --exit-code
                cargo xtask check-workflows
                python3 script/check-repository-inputs
                python3 script/test-repository-inputs.py
                cargo xtask products --check
                cargo xtask bundle --product rust \
                  --platform linux \
                  --target x86_64-unknown-linux-gnu \
                  --dry-run
            "#}))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

fn windows_updater_tests() -> NamedJob {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let product = manifest
        .product(RUST_PRODUCT_ID)
        .expect("Rust product must exist");
    named::job(
        Job::default()
            .runs_on("windows-2022")
            .timeout_minutes(45u32)
            .add_env(("RELEASE_VERSION", "1.2.3"))
            .add_env(("ZED_PRODUCT_DISPLAY_NAME", product.display_name.clone()))
            .add_env((
                "ZED_PRODUCT_WINDOWS_INSTALLER_ID",
                product.windows_installer_id.clone(),
            ))
            .add_env(("ZED_PRODUCT_ICON_SET", format!("${{{{ github.workspace }}}}/{}", product.icon_set)))
            .add_step(steps::enable_windows_long_paths())
            .add_step(steps::checkout_repo())
            .add_step(named::pwsh(indoc::indoc! {r#"
                $ErrorActionPreference = 'Stop'
                $PSNativeCommandUseErrorActionPreference = $true
                cargo test --locked -p auto_update_helper -p xtask -p windows_resources
                foreach ($version in @('1.2.3', '2.3.4-rc.2+build.7')) {
                    $env:RELEASE_VERSION = $version
                    cargo build --locked -p auto_update_helper
                    $metadata = (Get-Item target/debug/auto_update_helper.exe).VersionInfo
                    if ($metadata.ProductName -ne $env:ZED_PRODUCT_DISPLAY_NAME -or $metadata.FileVersion -ne $version -or $metadata.ProductVersion -ne $version) {
                        throw "Updater product metadata does not match the resolved release environment"
                    }
                    $numeric = ($version -split '[-+]')[0] -split '\.'
                    if ($metadata.FileMajorPart -ne [int]$numeric[0] -or $metadata.FileMinorPart -ne [int]$numeric[1] -or $metadata.FileBuildPart -ne [int]$numeric[2]) {
                        throw "Updater numeric version lost a semantic version component"
                    }
                }

                $innoSetupPath = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
                $installerResourceLines = @(
                    Get-Content crates/zed/resources/windows/zed.iss |
                        Where-Object { $_ -match '^(AppId|SetupIconFile)=' }
                )
                if ($installerResourceLines.Count -ne 2) {
                    throw "Expected the AppId and SetupIconFile directives in the Windows installer template"
                }
                $fixtureDirectory = Join-Path $env:RUNNER_TEMP 'inno-installer-resource-validation'
                New-Item -ItemType Directory -Path $fixtureDirectory -Force | Out-Null
                $fixturePath = Join-Path $fixtureDirectory 'installer-resources.iss'
                $productIconPath = Join-Path $env:ZED_PRODUCT_ICON_SET 'app-icon.ico'
                @(
                    '[Setup]'
                    $installerResourceLines
                    'AppName=Installer Identity Validation'
                    'AppVersion=1.0.0'
                    'DefaultDirName={tmp}\InstallerIdentityValidation'
                    'Uninstallable=no'
                ) | Set-Content -Path $fixturePath -Encoding utf8
                & $innoSetupPath '/O-' "/DAppId=$env:ZED_PRODUCT_WINDOWS_INSTALLER_ID" "/DProductIconFile=$productIconPath" $fixturePath
                if ($LASTEXITCODE -ne 0) {
                    throw "Inno Setup rejected the catalog-derived installer identity or icon"
                }
            "#})),
    )
}

fn windows_jvm_validation() -> NamedJob {
    named::job(
        Job::default()
            .runs_on("windows-2022")
            .timeout_minutes(90u32)
            .add_env(("CARGO_PROFILE_TEST_DEBUG", 0))
            .add_step(steps::enable_windows_long_paths())
            .add_step(steps::checkout_repo())
            .add_step(named::pwsh(
                "cargo test --locked -p fs --features test-support --test integration real_fs_confined_file_rejects_replaced_paths",
            ))
            .add_step(named::pwsh(
                "cargo test --locked -p project --features jvm-tools,test-support --test integration collects_windows_maven_report_and_rejects_parent_junction",
            ))
            .add_step(named::pwsh(
                "cargo test --locked -p onboarding --lib windows_path_jdk_probe_keeps_java_and_javac_in_one_installation",
            ))
            .add_step(named::pwsh(
                "cargo test --locked -p terminal_view --features jvm-tools,test-support --lib groovy_build_uses_existing_terminal_panel",
            ))
            .add_step(named::pwsh(
                "cargo test --locked -p tasks_ui --features jvm-test-actions --lib selected_java_windows_maven_case_uses_tests_panel_and_task_terminal",
            )),
    )
}

fn shared_validation(workers: &[&NamedJob]) -> NamedJob {
    let worker_names = workers
        .iter()
        .map(|worker| worker.name.clone())
        .collect::<Vec<_>>();
    let mut check_results_script = String::from("exit_code=0\n");
    for worker in workers {
        check_results_script.push_str(&format!(
            "if [[ \"${{{}_RESULT}}\" != \"success\" ]]; then exit_code=1; fi\n",
            worker.name.to_uppercase()
        ));
    }
    check_results_script.push_str("exit \"$exit_code\"\n");
    let check_results = workers
        .iter()
        .fold(named::bash(check_results_script), |step, worker| {
            step.add_env((
                format!("{}_RESULT", worker.name.to_uppercase()),
                format!("${{{{ needs.{}.result }}}}", worker.name),
            ))
        });

    named::job(
        Job::default()
            .needs(worker_names)
            .cond(Expression::new("${{ always() }}"))
            .runs_on("ubuntu-22.04")
            .timeout_minutes(5u32)
            .add_step(check_results),
    )
}

fn sim_backend_validation() -> NamedJob {
    let include = vec![
        json!({
            "platform": "linux",
            "runner": "ubuntu-22.04",
        }),
        json!({
            "platform": "macos",
            "runner": "macos-15",
        }),
        json!({
            "platform": "windows",
            "runner": "windows-2022",
        }),
    ];

    named::job(
        Job::default()
            .runs_on("${{ matrix.runner }}")
            .timeout_minutes(60u32)
            .strategy(
                Strategy::default()
                    .fail_fast(false)
                    .matrix(json!({ "include": include })),
            )
            .add_step(
                steps::enable_windows_long_paths()
                    .if_condition(Expression::new("matrix.platform == 'windows'")),
            )
            .add_step(steps::checkout_repo())
            .add_step(
                steps::setup_cargo_config(Platform::Linux)
                    .if_condition(Expression::new("matrix.platform == 'linux'")),
            )
            .add_step(
                steps::setup_cargo_config(Platform::Mac)
                    .if_condition(Expression::new("matrix.platform == 'macos'")),
            )
            .add_step(
                steps::setup_cargo_config(Platform::Windows)
                    .if_condition(Expression::new("matrix.platform == 'windows'")),
            )
            .add_step(
                named::bash(
                    "cargo clippy --release --all-targets --all-features -p sim_backend_corex -p sim_backend_cuda -p sim_backend_directml -p sim_backend_metal -p sim_backend_mlu -p sim_backend_npu -p sim_backend_rocm -p sim_backend_xpu -- --deny warnings",
                )
                .if_condition(Expression::new("matrix.platform == 'linux'")),
            )
            .add_step(
                named::bash(
                    "cargo clippy --release --all-targets --all-features -p sim_backend_metal -- --deny warnings",
                )
                .if_condition(Expression::new("matrix.platform == 'macos'")),
            )
            .add_step(
                named::pwsh(
                    "cargo clippy --release --all-targets --all-features -p sim_backend_cuda -p sim_backend_directml -- --deny warnings",
                )
                .if_condition(Expression::new("matrix.platform == 'windows'")),
            )
            .add_step(
                steps::cleanup_cargo_config(Platform::Linux)
                    .if_condition(Expression::new("always() && matrix.platform == 'linux'")),
            )
            .add_step(
                steps::cleanup_cargo_config(Platform::Mac)
                    .if_condition(Expression::new("always() && matrix.platform == 'macos'")),
            )
            .add_step(
                steps::cleanup_cargo_config(Platform::Windows)
                    .if_condition(Expression::new("always() && matrix.platform == 'windows'")),
            ),
    )
}

fn product_smoke(validation: &NamedJob) -> NamedJob {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let include = manifest
        .enabled_products()
        .map(|product| {
            json!({
                "product": product.id,
                "application_features": product.cargo_features.join(","),
                "remote_features": product.remote_server_features.join(","),
            })
        })
        .collect::<Vec<_>>();

    named::job(
        Job::default()
            .needs([validation.name.clone()])
            .runs_on("ubuntu-22.04")
            .timeout_minutes(45u32)
            .strategy(Strategy::default().fail_fast(false).matrix(json!({ "include": include })))
            .map(use_clang)
            .add_step(steps::checkout_repo())
            .add_step(steps::setup_cargo_config(Platform::Linux))
            .map(steps::install_linux_dependencies)
            .add_step(named::bash(
                "cargo xtask bundle --product \"$PRODUCT_ID\" --platform linux --target x86_64-unknown-linux-gnu --dry-run",
            ).add_env(("PRODUCT_ID", "${{ matrix.product }}")))
            .add_step(named::bash(
                "cargo check --release --package zed --no-default-features --features \"$APPLICATION_FEATURES\"",
            ).add_env(("ZED_PRODUCT_ID", "${{ matrix.product }}")).add_env(("APPLICATION_FEATURES", "${{ matrix.application_features }}")))
            .add_step(named::bash(
                "cargo check --release --package remote_server --no-default-features --features \"$REMOTE_FEATURES\"",
            ).add_env(("ZED_PRODUCT_ID", "${{ matrix.product }}")).add_env(("REMOTE_FEATURES", "${{ matrix.remote_features }}")))
            .add_step(steps::cleanup_cargo_config(Platform::Linux)),
    )
}

pub fn orchestrate_for_extension(rules: &[&PathCondition]) -> NamedJob {
    orchestrate_impl(rules)
}

fn orchestrate_impl(rules: &[&PathCondition]) -> NamedJob {
    let name = "orchestrate".to_owned();
    let step_name = "filter".to_owned();
    let mut script = String::new();

    script.push_str(indoc::indoc! {r#"
        set -euo pipefail
        if [ -z "$GITHUB_BASE_REF" ]; then
          echo "Not in a PR context (i.e., push to main/stable/preview)"
          COMPARE_REV="$(git rev-parse HEAD~1)"
        else
          echo "In a PR context comparing to pull_request.base.ref"
          git fetch origin "$GITHUB_BASE_REF" --depth=350
          COMPARE_REV="$(git merge-base "origin/${GITHUB_BASE_REF}" HEAD)"
        fi
        CHANGED_FILES="$(git diff --name-only "$COMPARE_REV" "$GITHUB_SHA")"

    "#});

    script.push_str(indoc::indoc! {r#"
        # When running from a subdirectory, git diff returns repo-root-relative paths.
        # Filter to only files within the current working directory and strip the prefix.
        REPO_SUBDIR="$(git rev-parse --show-prefix)"
        REPO_SUBDIR="${REPO_SUBDIR%/}"
        if [ -n "$REPO_SUBDIR" ]; then
            CHANGED_FILES="$(echo "$CHANGED_FILES" | grep "^${REPO_SUBDIR}/" | sed "s|^${REPO_SUBDIR}/||" || true)"
        fi

    "#});

    script.push_str(indoc::indoc! {r#"
        check_pattern() {
          local output_name="$1"
          local pattern="$2"
          local grep_arg="$3"

          echo "$CHANGED_FILES" | grep "$grep_arg" "$pattern" && \
            echo "${output_name}=true" >> "$GITHUB_OUTPUT" || \
            echo "${output_name}=false" >> "$GITHUB_OUTPUT"
        }

    "#});

    let mut outputs = IndexMap::new();

    for rule in rules {
        assert!(
            rule.set_by_step
                .borrow_mut()
                .replace(name.clone())
                .is_none()
        );
        assert!(
            outputs
                .insert(
                    rule.name.to_owned(),
                    format!("${{{{ steps.{}.outputs.{} }}}}", step_name, rule.name)
                )
                .is_none()
        );

        let grep_arg = if rule.invert { "-qvP" } else { "-qP" };
        script.push_str(&format!(
            "check_pattern \"{}\" '{}' {}\n",
            rule.name, rule.pattern, grep_arg
        ));
    }

    let job = Job::default()
        .runs_on(runners::LINUX_SMALL)
        .with_repository_owner_guard()
        .outputs(outputs)
        .add_step(steps::checkout_repo().with_deep_history_on_non_main())
        .add_step(Step::new(step_name.clone()).run(script).id(step_name));

    NamedJob { name, job }
}

pub fn tests_pass(jobs: &[NamedJob], extra_job_names: &[&str]) -> NamedJob {
    let mut script = String::from(indoc::indoc! {r#"
        set +x
        EXIT_CODE=0

        check_result() {
          echo "* $1: $2"
          if [[ "$2" != "skipped" && "$2" != "success" ]]; then EXIT_CODE=1; fi
        }

    "#});

    let all_names: Vec<&str> = jobs
        .iter()
        .map(|job| job.name.as_str())
        .chain(extra_job_names.iter().copied())
        .collect();

    let env_entries: Vec<_> = all_names
        .iter()
        .map(|name| {
            let env_name = format!("RESULT_{}", name.to_uppercase());
            let env_value = format!("${{{{ needs.{}.result }}}}", name);
            (env_name, env_value)
        })
        .collect();

    script.push_str(
        &all_names
            .iter()
            .zip(env_entries.iter())
            .map(|(name, (env_name, _))| format!("check_result \"{}\" \"${}\"", name, env_name))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    script.push_str("\n\nexit $EXIT_CODE\n");

    let job = Job::default()
        .runs_on(runners::LINUX_SMALL)
        .needs(
            all_names
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<String>>(),
        )
        .cond(repository_owner_guard_expression(true))
        .add_step(
            env_entries
                .into_iter()
                .fold(named::bash(&script), |step, env_item| {
                    step.add_env(env_item)
                }),
        );

    named::job(job)
}

/// Bash script snippet that detects changed extension directories from `$CHANGED_FILES`.
/// Assumes `$CHANGED_FILES` is already set. Sets `$EXTENSIONS_JSON` to a JSON array of
/// changed extension paths. Callers are responsible for writing the result to `$GITHUB_OUTPUT`.
pub(crate) const DETECT_CHANGED_EXTENSIONS_SCRIPT: &str = indoc::indoc! {r#"
    # Detect changed extension directories (excluding extensions/workflows)
    CHANGED_EXTENSIONS=$(echo "$CHANGED_FILES" | grep -oP '^extensions/[^/]+(?=/)' | sort -u | grep -v '^extensions/workflows$' || true)
    # Filter out deleted extensions
    EXISTING_EXTENSIONS=""
    for ext in $CHANGED_EXTENSIONS; do
        if [ -f "$ext/extension.toml" ]; then
            EXISTING_EXTENSIONS=$(printf '%s\n%s' "$EXISTING_EXTENSIONS" "$ext")
        fi
    done
    CHANGED_EXTENSIONS=$(echo "$EXISTING_EXTENSIONS" | sed '/^$/d')
    if [ -n "$CHANGED_EXTENSIONS" ]; then
        EXTENSIONS_JSON=$(echo "$CHANGED_EXTENSIONS" | jq -R -s -c 'split("\n") | map(select(length > 0))')
    else
        EXTENSIONS_JSON="[]"
    fi
"#};

const TS_QUERY_LS_FILE: &str = "ts_query_ls-x86_64-unknown-linux-gnu.tar.gz";
const CI_TS_QUERY_RELEASE: &str = "tags/v3.15.1";

pub(crate) fn fetch_ts_query_ls() -> Step<Use> {
    named::uses(
        "dsaltares",
        "fetch-gh-release-asset",
        "aa37ae5c44d3c9820bc12fe675e8670ecd93bd1c",
    ) // v1.1.1
    .add_with(("repo", "ribru17/ts_query_ls"))
    .add_with(("version", CI_TS_QUERY_RELEASE))
    .add_with(("file", TS_QUERY_LS_FILE))
}

pub(crate) enum RunContext {
    Extension,
}

pub(crate) fn run_ts_query_ls(context: RunContext) -> Step<Run> {
    named::bash(formatdoc!(
        r#"tar -xf "$GITHUB_WORKSPACE/{TS_QUERY_LS_FILE}" -C "$GITHUB_WORKSPACE"
        "$GITHUB_WORKSPACE/ts_query_ls" format --check {directory} || {{
            echo "Found unformatted queries, please format them with ts_query_ls."
            echo "For easy use, install the Tree-sitter query extension:"
            echo "zed://extension/tree-sitter-query"
            false
        }}"#,
        directory = match context {
            RunContext::Extension => "languages",
        }
    ))
}

pub(crate) fn run_platform_tests_no_filter(platform: Platform) -> NamedJob {
    run_platform_tests_impl(platform, false, false)
}

fn run_platform_tests_impl(platform: Platform, filter_packages: bool, harden: bool) -> NamedJob {
    let runner = match platform {
        Platform::Windows => runners::WINDOWS_DEFAULT,
        Platform::Linux => runners::LINUX_DEFAULT,
        Platform::Mac => runners::MAC_DEFAULT,
    };
    NamedJob {
        name: format!("run_tests_{platform}"),
        job: release_job(&[])
            .runs_on(runner)
            .when(platform == Platform::Linux, |job| {
                job.add_service(
                    "postgres",
                    Container::new("postgres:15@sha256:1b92e7a80c021647bf70f5d3eb66066a998e4f5cf43c07bb9dc9f729782cf88e")
                        .add_env(("POSTGRES_HOST_AUTH_METHOD", "trust"))
                        .ports(vec![Port::Name("5432:5432".into())])
                        .options(
                            "--health-cmd pg_isready \
                             --health-interval 500ms \
                             --health-timeout 5s \
                             --health-retries 10",
                        ),
                )
            })
            .when(harden && platform == Platform::Linux, |this| {
                this.add_step(steps::harden_runner())
            })
            .add_step(steps::checkout_repo())
            .add_step(steps::setup_cargo_config(platform))
            .when(platform == Platform::Mac, |this| {
                this.add_step(steps::cache_rust_dependencies_namespace())
            })
            .when(platform == Platform::Linux, |this| {
                use_clang(this.add_step(steps::cache_rust_dependencies_namespace()))
            })
            .when(
                platform == Platform::Linux,
                steps::install_linux_dependencies,
            )
            .add_step(steps::setup_node())
            .when(
                platform == Platform::Linux || platform == Platform::Mac,
                |job| job.add_step(steps::cargo_install_nextest()),
            )
            .add_step(steps::clear_target_dir_if_large(platform))
            .add_step(steps::setup_sccache(platform))
            .when(filter_packages, |job| {
                job.add_step(
                    steps::cargo_nextest(platform).with_changed_packages_filter("orchestrate"),
                )
            })
            .when(!filter_packages, |job| {
                job.add_step(steps::cargo_nextest(platform))
            })
            .add_step(steps::show_sccache_stats(platform))
            .add_step(steps::cleanup_cargo_config(platform)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;

    fn job<'a>(workflow: &'a Value, name: &str) -> &'a Value {
        workflow
            .get("jobs")
            .and_then(|jobs| jobs.get(name))
            .unwrap_or_else(|| panic!("missing {name} job"))
    }

    fn run_commands(job: &Value) -> Vec<&str> {
        job.get("steps")
            .and_then(Value::as_sequence)
            .expect("job steps")
            .iter()
            .filter_map(|step| step.get("run").and_then(Value::as_str))
            .collect()
    }

    fn needs(job: &Value) -> Vec<&str> {
        job.get("needs")
            .and_then(Value::as_sequence)
            .map(|needs| needs.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default()
    }

    #[test]
    fn full_linux_validation_prepares_disk_before_build_dependencies() -> anyhow::Result<()> {
        let workflow = run_tests()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize CI workflow: {error:?}"))?;
        let parsed: Value = serde_yaml::from_str(&workflow)?;
        for (name, timeout) in [
            ("shared_clippy", 60),
            ("project_benchmarks", 75),
            ("rust_tools_validation", 45),
            ("copper_tests", 90),
            ("graphics_feature_validation", 120),
            ("multiplayer_tools_standard", 120),
            ("multiplayer_tools_multiplayer", 120),
        ] {
            let steps = job(&parsed, name)
                .get("steps")
                .and_then(Value::as_sequence)
                .expect("Linux validation steps");
            let cleanup = steps
                .iter()
                .position(|step| {
                    step.get("name").and_then(Value::as_str) == Some("steps::free_linux_disk_space")
                })
                .expect("existing Linux disk preparation");
            let setup = steps
                .iter()
                .position(|step| {
                    step.get("name").and_then(Value::as_str) == Some("steps::setup_linux")
                })
                .expect("Linux dependency setup");
            assert!(
                cleanup < setup,
                "{name} must prepare disk before installing build dependencies"
            );
            assert_eq!(
                job(&parsed, name).get("runs-on").and_then(Value::as_str),
                Some("ubuntu-22.04")
            );
            assert_eq!(
                job(&parsed, name)
                    .get("timeout-minutes")
                    .and_then(Value::as_u64),
                Some(timeout)
            );
        }
        Ok(())
    }

    #[test]
    fn windows_long_paths_are_enabled_before_checkout() -> anyhow::Result<()> {
        let workflow = run_tests()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize CI workflow: {error:?}"))?;
        let parsed: Value = serde_yaml::from_str(&workflow)?;
        for name in [
            "windows_updater_tests",
            "windows_jvm_validation",
            "sim_backend_validation",
        ] {
            let steps = job(&parsed, name)
                .get("steps")
                .and_then(Value::as_sequence)
                .expect("Windows job steps");
            let configuration = steps
                .iter()
                .position(|step| {
                    step.get("run").and_then(Value::as_str)
                        == Some("git config --global core.longpaths true")
                })
                .expect("Windows long-path configuration");
            let checkout = steps
                .iter()
                .position(|step| {
                    step.get("uses")
                        .and_then(Value::as_str)
                        .is_some_and(|action| action.starts_with("actions/checkout@"))
                })
                .expect("repository checkout");
            assert!(
                configuration < checkout,
                "{name} configures long paths after checkout"
            );
            assert_eq!(
                steps[configuration].get("shell").and_then(Value::as_str),
                Some("pwsh")
            );
            if name == "sim_backend_validation" {
                assert_eq!(
                    steps[configuration].get("if").and_then(Value::as_str),
                    Some("matrix.platform == 'windows'")
                );
            }
        }
        Ok(())
    }

    #[test]
    fn copper_compiler_cache_preserves_required_checks_and_budget() -> anyhow::Result<()> {
        let workflow = run_tests()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize CI workflow: {error:?}"))?;
        let parsed: Value = serde_yaml::from_str(&workflow)?;
        let copper = job(&parsed, "copper_tests");
        assert_eq!(
            copper.get("timeout-minutes").and_then(Value::as_u64),
            Some(90)
        );
        let environment = copper.get("env").expect("Copper environment");
        for (name, value) in [
            ("SCCACHE_GHA_ENABLED", "on"),
            ("RUSTC_WRAPPER", "sccache"),
            ("CARGO_PROFILE_TEST_DEBUG", "0"),
        ] {
            assert_eq!(environment.get(name).and_then(Value::as_str), Some(value));
        }
        for secret in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"] {
            assert!(environment.get(secret).is_none());
        }
        let steps = copper
            .get("steps")
            .and_then(Value::as_sequence)
            .expect("Copper steps");
        let cache = steps
            .iter()
            .position(|step| {
                step.get("uses").and_then(Value::as_str)
                    == Some(
                        "mozilla-actions/sccache-action@fc920bf0ec8de6ee65d409111f7ec508035751ba",
                    )
            })
            .expect("existing pinned compiler cache action");
        let namespace = steps
            .iter()
            .position(|step| {
                step.get("run")
                    .and_then(Value::as_str)
                    .is_some_and(|command| {
                        command.contains("git hash-object rust-toolchain.toml")
                            && command.contains(
                                "copper-tests-v1-${RUNNER_OS}-${RUNNER_ARCH}-${TOOLCHAIN_HASH}",
                            )
                    })
            })
            .expect("platform and toolchain compiler-cache namespace");
        let regular_command = rust_product_nextest_command(false);
        let foreground_command = rust_product_nextest_command(true);
        let regular = steps
            .iter()
            .position(|step| {
                step.get("run").and_then(Value::as_str) == Some(regular_command.as_str())
            })
            .expect("unchanged required product tests");
        let foreground = steps
            .iter()
            .position(|step| {
                step.get("run").and_then(Value::as_str) == Some(foreground_command.as_str())
            })
            .expect("unchanged foreground budget checks");
        let finalization = steps
            .iter()
            .position(|step| {
                step.get("run")
                    .and_then(Value::as_str)
                    .is_some_and(|command| {
                        command.contains("sccache --show-stats")
                            && command.contains("sccache --stop-server")
                    })
            })
            .expect("compiler cache statistics and shutdown");
        assert!(
            cache < namespace
                && namespace < regular
                && regular < foreground
                && foreground < finalization
        );
        assert_eq!(
            steps[finalization].get("if"),
            Some(&serde_yaml::to_value(Expression::new("always()"))?)
        );
        assert_eq!(
            steps
                .iter()
                .filter(|step| step
                    .get("run")
                    .and_then(Value::as_str)
                    .is_some_and(|command| command.starts_with("cargo nextest run")))
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    fn rust_product_ci_has_shared_validation_and_one_smoke_row() -> anyhow::Result<()> {
        let workflow = run_tests()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize CI workflow: {error:?}"))?;
        let parsed: Value = serde_yaml::from_str(&workflow)?;

        let worker_names = [
            "shared_clippy",
            "copper_tests",
            "graphics_feature_validation",
            "project_benchmarks",
            "rust_tools_validation",
            "release_automation_validation",
            "windows_updater_tests",
            "windows_jvm_validation",
            "multiplayer_tools_standard",
            "multiplayer_tools_multiplayer",
        ];
        for worker_name in worker_names {
            assert!(needs(job(&parsed, worker_name)).is_empty());
        }

        let graphics_commands = run_commands(job(&parsed, "graphics_feature_validation"));
        let product_boundaries = graphics_commands
            .iter()
            .find(|command| command.contains("graphics_surface_respects_sim_feature"))
            .expect("graphics profile checks");
        assert!(product_boundaries.contains("-p graphics_core -p graphics_ui --lib"));
        assert!(product_boundaries.contains("-p graphics_core --test native_worker"));
        assert!(product_boundaries.contains("-p gpui --lib tab_stop::tests::"));
        assert!(product_boundaries.contains("--features agentic-tools,sim"));
        assert!(product_boundaries.contains("--features agentic-tools,jvm-tools"));
        assert!(product_boundaries.contains("ZED_PRODUCT_DEV_PREVIEW=1 ZED_RELEASE_CHANNEL=dev"));
        assert!(product_boundaries.contains("--bin zed native_sim_"));

        let shared_validation = job(&parsed, "shared_validation");
        assert_eq!(needs(shared_validation), worker_names);
        assert_eq!(
            shared_validation.get("if").and_then(Value::as_str),
            Some("${{ always() }}")
        );
        let aggregator_commands = run_commands(shared_validation);
        assert_eq!(aggregator_commands.len(), 1);
        assert!(aggregator_commands[0].contains("exit \"$exit_code\""));
        let aggregator_steps = shared_validation
            .get("steps")
            .and_then(Value::as_sequence)
            .expect("aggregator steps");
        assert_eq!(aggregator_steps.len(), 1);
        let aggregator_env = aggregator_steps[0]
            .get("env")
            .expect("aggregator result environment");
        for worker_name in worker_names {
            let environment_name = format!("{}_RESULT", worker_name.to_uppercase());
            let expected_result = format!("${{{{ needs.{worker_name}.result }}}}");
            assert_eq!(
                aggregator_env
                    .get(&environment_name)
                    .and_then(Value::as_str),
                Some(expected_result.as_str())
            );
            assert!(
                aggregator_commands[0]
                    .contains(&format!("[[ \"${{{environment_name}}}\" != \"success\" ]]"))
            );
        }

        assert_eq!(needs(job(&parsed, "product_smoke")), ["shared_validation"]);
        for profile in ["standard", "multiplayer"] {
            let profile_job_name = format!("multiplayer_tools_{profile}");
            let profile_job = job(&parsed, &profile_job_name);
            let expected_command =
                format!("./script/check-multiplayer-tools --full --profile {profile}");
            let commands = run_commands(profile_job);
            assert_eq!(
                commands
                    .iter()
                    .filter(|command| command.contains("check-multiplayer-tools"))
                    .copied()
                    .collect::<Vec<_>>(),
                [expected_command.as_str()]
            );
            assert_eq!(
                profile_job
                    .get("env")
                    .and_then(|environment| environment.get("ZED_PRODUCT_ID"))
                    .and_then(Value::as_str),
                Some(RUST_PRODUCT_ID)
            );
            assert!(needs(profile_job).is_empty());
        }

        let shared_clippy_commands = run_commands(job(&parsed, "shared_clippy"));
        assert!(shared_clippy_commands.contains(&"cargo fmt --all -- --check"));
        assert!(shared_clippy_commands.contains(&"./script/clippy"));
        assert!(
            shared_clippy_commands
                .contains(&"python3 script/test-collaborative-workspace-inventory.py")
        );
        let shared_clippy_steps = job(&parsed, "shared_clippy")
            .get("steps")
            .and_then(Value::as_sequence)
            .expect("shared validation should have steps");
        let python_setup_position = shared_clippy_steps
            .iter()
            .position(|step| {
                step.get("uses")
                    .and_then(Value::as_str)
                    .is_some_and(|action| action.starts_with("actions/setup-python@"))
                    && step
                        .get("with")
                        .and_then(|inputs| inputs.get("python-version"))
                        .and_then(Value::as_str)
                        == Some("3.12")
            })
            .expect("inventory validation should use Python 3.12");
        let inventory_check_position = shared_clippy_steps
            .iter()
            .position(|step| {
                step.get("run")
                    .and_then(Value::as_str)
                    .is_some_and(|command| {
                        command
                            .contains("git cat-file -e HEAD:projects/buzz/Cargo.toml 2>/dev/null")
                            && command.contains(
                                "./script/check-collaborative-workspace-inventory --catalog all",
                            )
                            && command.contains("full inventory audit deferred")
                    })
            })
            .expect("inventory validation should run");
        assert!(python_setup_position < inventory_check_position);
        let copper_test_commands = run_commands(job(&parsed, "copper_tests"));
        let copper_disk_cleanup = copper_test_commands
            .iter()
            .position(|command| {
                command.contains("sudo swapoff -a")
                    && command.contains("/opt/hostedtoolcache")
                    && command.contains("/usr/local/share/boost")
                    && command.contains("/usr/share/swift")
                    && command.contains("/opt/az")
                    && command.contains("/usr/share/miniconda")
                    && command.contains("/mnt/swapfile")
            })
            .expect("Rust product tests should reclaim unused runner disk");
        let product_nextest_command = rust_product_nextest_command(false);
        let copper_nextest = copper_test_commands
            .iter()
            .position(|command| *command == product_nextest_command)
            .expect("Rust product nextest command");
        assert!(copper_disk_cleanup < copper_nextest);
        assert_eq!(
            copper_test_commands
                .iter()
                .filter(|command| command.starts_with("cargo nextest run"))
                .count(),
            2
        );
        assert!(copper_test_commands.contains(&rust_product_nextest_command(true).as_str()));
        let measurement = rust_product_nextest_command(true);
        assert!(measurement.contains("--test-threads 1 --success-output immediate"));
        assert!(measurement.contains("--no-tests=fail"));
        assert!(measurement.contains("cargo_dashboard_foreground_budget"));
        assert!(measurement.contains("cargo_dashboard_production_foreground_budget"));
        assert!(
            measurement.contains("test_explorer::tests::structured_execution_foreground_budget")
        );
        assert!(
            product_nextest_command
                .contains("test_explorer::tests::structured_execution_foreground_budget")
        );
        for command in [&measurement, &product_nextest_command] {
            assert!(command.contains(
                "test(=profile_view::tests::profile_view_deep_selection_expands_ancestors_in_one_batch)"
            ));
        }
        assert!(!measurement.contains("not ("));
        assert!(!measurement.contains("--retries"));
        assert!(
            copper_test_commands
                .iter()
                .position(|command| *command == measurement)
                .expect("measurement step")
                > copper_nextest
        );
        assert_eq!(
            job(&parsed, "copper_tests")
                .get("env")
                .and_then(|environment| environment.get("CARGO_PROFILE_TEST_DEBUG"))
                .and_then(Value::as_str),
            Some("0")
        );
        assert_eq!(
            job(&parsed, "copper_tests")
                .get("env")
                .and_then(|environment| environment.get("ZED_PRODUCT_ID"))
                .and_then(Value::as_str),
            Some(RUST_PRODUCT_ID)
        );
        assert_eq!(
            job(&parsed, "copper_tests")
                .get("env")
                .and_then(|environment| environment.get("ZED_PRODUCT_UPDATE_BASE_URL"))
                .and_then(Value::as_str),
            Some(RUST_PRODUCT_TEST_UPDATE_BASE_URL)
        );
        let product_nextest_arguments = product_nextest_command
            .split_whitespace()
            .collect::<Vec<_>>();
        for package in RUST_PRODUCT_TEST_PACKAGES {
            assert!(
                product_nextest_arguments
                    .windows(2)
                    .any(|arguments| arguments == ["-p", package]),
                "Rust product tests must select {package}"
            );
        }
        for baseline_collaboration_package in ["client", "call", "channel", "collab_ui"] {
            assert!(
                product_nextest_arguments
                    .windows(2)
                    .any(|arguments| { arguments == ["-p", baseline_collaboration_package] }),
                "baseline desktop collaboration package {baseline_collaboration_package} must remain required"
            );
        }
        assert!(
            !product_nextest_arguments
                .windows(2)
                .any(|arguments| arguments == ["-p", "collab"])
        );
        assert!(
            !product_nextest_arguments
                .windows(2)
                .any(|arguments| { arguments[0] == "-p" && arguments[1].starts_with("sim_") })
        );
        assert!(!product_nextest_command.contains("multiplayer-tools"));
        assert!(
            product_nextest_command
                .contains("--features zed/agentic-tools,zed/rust-tools,remote_server/rust-tools")
        );
        for worker_name in [
            "shared_clippy",
            "project_benchmarks",
            "rust_tools_validation",
            "release_automation_validation",
        ] {
            assert!(
                job(&parsed, worker_name)
                    .get("env")
                    .and_then(|environment| environment.get("CARGO_PROFILE_TEST_DEBUG"))
                    .is_none()
            );
        }
        let benchmark_commands = run_commands(job(&parsed, "project_benchmarks"));
        let benchmark_strategy = job(&parsed, "project_benchmarks")
            .get("strategy")
            .expect("benchmark matrix strategy");
        assert_eq!(
            benchmark_strategy.get("fail-fast").and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            benchmark_strategy
                .get("matrix")
                .and_then(|matrix| matrix.get("include")),
            Some(&serde_yaml::to_value(json!([
                { "benchmark": "cargo_workspace" },
                { "benchmark": "structured_execution" },
            ]))?)
        );
        let benchmark_steps = job(&parsed, "project_benchmarks")
            .get("steps")
            .and_then(Value::as_sequence)
            .expect("benchmark steps");
        for (feature, benchmark) in [
            ("cargo-workspace", "cargo_workspace"),
            ("structured-execution", "structured_execution"),
        ] {
            let command = format!(
                "cargo bench -p project --features {feature} --bench {benchmark} -- --noplot"
            );
            let step = benchmark_steps
                .iter()
                .find(|step| step.get("run").and_then(Value::as_str) == Some(command.as_str()))
                .expect("isolated benchmark command");
            assert_eq!(
                step.get("if").and_then(Value::as_str),
                Some(format!("matrix.benchmark == '{benchmark}'").as_str())
            );
        }
        assert_eq!(
            job(&parsed, "project_benchmarks")
                .get("timeout-minutes")
                .and_then(Value::as_u64),
            Some(75)
        );
        assert!(
            benchmark_commands
                .iter()
                .any(|command| command.contains("--bench cargo_workspace -- --noplot"))
        );
        assert!(
            benchmark_commands
                .iter()
                .any(|command| command.contains("--bench structured_execution -- --noplot"))
        );
        let rust_tools_commands = run_commands(job(&parsed, "rust_tools_validation"));
        let rust_tools_fetch = rust_tools_commands
            .iter()
            .position(|command| *command == "cargo fetch --locked")
            .expect("rust-tools dependencies should be fetched on the fresh runner");
        let rust_tools_offline_validation = rust_tools_commands
            .iter()
            .position(|command| {
                *command == "./script/test-rust-tools-environments --matrix --offline"
            })
            .expect("rust-tools offline validation should be generated");
        let native_task_helper_validation = rust_tools_commands
            .iter()
            .position(|command| {
                *command == "cargo test --locked --offline -p native_task_helper --lib"
            })
            .expect("Linux native task helper process fixtures should execute");
        assert!(rust_tools_fetch < native_task_helper_validation);
        assert!(native_task_helper_validation < rust_tools_offline_validation);

        let release_validation_commands =
            run_commands(job(&parsed, "release_automation_validation"));
        assert_eq!(release_validation_commands.len(), 3);
        assert!(
            !release_validation_commands
                .iter()
                .any(|command| command.contains("./script/linux"))
        );
        let release_validation = release_validation_commands
            .iter()
            .find(|command| command.contains("cargo test -p xtask"))
            .expect("release pipeline validation command");
        for required_command in [
            "python3 -m unittest discover -s script -p 'test_refresh_website.py'",
            "python3 -m unittest discover -s script -p 'test_product_release.py'",
            "cargo xtask workflows",
            "git diff --exit-code",
            "cargo xtask check-workflows",
            "cargo xtask products --check",
            "cargo xtask bundle --product rust",
            "--dry-run",
        ] {
            assert!(
                release_validation.contains(required_command),
                "release validation must run {required_command}"
            );
        }

        assert_eq!(workflow.matches("runs-on:").count(), 13);
        let updater = job(&parsed, "windows_updater_tests");
        assert_eq!(
            updater.get("runs-on").and_then(Value::as_str),
            Some("windows-2022")
        );
        assert_eq!(
            updater.get("timeout-minutes").and_then(Value::as_u64),
            Some(45),
            "cold Windows updater builds must have enough time to reach the assertions"
        );
        assert!(
            run_commands(updater)
                .iter()
                .any(|command| command.contains("cargo test --locked -p auto_update_helper"))
        );
        assert!(run_commands(updater).iter().any(|command| {
            command.contains("Inno Setup rejected the catalog-derived installer identity or icon")
                && command.contains("SetupIconFile")
        }));
        let windows_jvm = job(&parsed, "windows_jvm_validation");
        assert_eq!(
            windows_jvm.get("runs-on").and_then(Value::as_str),
            Some("windows-2022")
        );
        assert!(run_commands(windows_jvm).iter().any(|command| {
            command.contains(
                "cargo test --locked -p fs --features test-support --test integration real_fs_confined_file_rejects_replaced_paths",
            )
        }));
        assert!(run_commands(windows_jvm).iter().any(|command| {
            command.contains("collects_windows_maven_report_and_rejects_parent_junction")
        }));
        assert!(run_commands(windows_jvm).iter().any(|command| {
            command.contains("windows_path_jdk_probe_keeps_java_and_javac_in_one_installation")
        }));
        assert!(
            run_commands(windows_jvm)
                .iter()
                .any(|command| { command.contains("groovy_build_uses_existing_terminal_panel") })
        );
        assert!(run_commands(windows_jvm).iter().any(|command| {
            command.contains("selected_java_windows_maven_case_uses_tests_panel_and_task_terminal")
        }));
        assert!(workflow.contains("runs-on: ubuntu-22.04"));
        assert!(workflow.contains("runner: macos-15"));
        assert!(workflow.contains("runner: windows-2022"));
        assert!(workflow.contains("workflow_dispatch: {}"));
        assert!(workflow.contains("cargo fmt --all -- --check"));
        assert!(workflow.contains("./script/clippy"));
        assert!(workflow.contains(&product_nextest_command));
        assert!(!workflow.contains("cargo nextest run --workspace"));
        assert!(workflow.contains("--bench cargo_workspace -- --noplot"));
        assert!(workflow.contains("--bench structured_execution -- --noplot"));
        assert!(workflow.contains("test-rust-tools-environments --matrix --offline"));
        assert!(workflow.contains("application_features: agentic-tools,rust-tools"));
        assert!(workflow.contains("remote_features: rust-tools"));
        assert!(workflow.contains("cargo xtask bundle --product"));
        assert_eq!(workflow.matches("cargo fmt --all -- --check").count(), 1);
        assert_eq!(workflow.matches("run: ./script/clippy").count(), 1);
        assert_eq!(workflow.matches(&product_nextest_command).count(), 1);
        assert_eq!(workflow.matches("/opt/hostedtoolcache").count(), 7);
        assert_eq!(
            workflow
                .matches("--bench cargo_workspace -- --noplot")
                .count(),
            1
        );
        assert_eq!(
            workflow
                .matches("--bench structured_execution -- --noplot")
                .count(),
            1
        );
        assert_eq!(
            workflow
                .matches("test-rust-tools-environments --matrix --offline")
                .count(),
            1
        );
        assert!(workflow.contains("-p sim_backend_corex"));
        assert!(workflow.contains("-p sim_backend_cuda"));
        assert!(workflow.contains("-p sim_backend_directml"));
        assert!(workflow.contains("-p sim_backend_metal"));
        assert!(workflow.contains("-p sim_backend_mlu"));
        assert!(workflow.contains("-p sim_backend_npu"));
        assert!(workflow.contains("-p sim_backend_rocm"));
        assert!(workflow.contains("-p sim_backend_xpu"));
        assert_eq!(workflow.matches("--all-targets --all-features").count(), 3);
        assert_eq!(workflow.matches("-- --deny warnings").count(), 3);
        assert_eq!(
            workflow
                .matches("git config --global core.longpaths true")
                .count(),
            3
        );
        let windows_long_paths = workflow
            .find("git config --global core.longpaths true")
            .expect("Windows long-path setup should be generated");
        let windows_backend_clippy = workflow
            .find("-p sim_backend_cuda -p sim_backend_directml")
            .expect("Windows backend Clippy command should be generated");
        assert!(windows_long_paths < windows_backend_clippy);
        assert!(workflow.contains("cancel-in-progress: true"));
        assert!(!workflow.contains("cargo clean"));

        for forbidden in [
            "repository_owner",
            "namespace-profile",
            "self-32vcpu",
            "R2_ACCOUNT_ID",
            "SENTRY_AUTH_TOKEN",
            "SLACK_WEBHOOK",
            "postgres:",
            "Rustlings",
            "product: jvm",
            "product: game",
        ] {
            assert!(!workflow.contains(forbidden), "found {forbidden} in CI");
        }

        Ok(())
    }
}
