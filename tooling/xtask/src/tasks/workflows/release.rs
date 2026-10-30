use gh_workflow::{
    Concurrency, Event, Expression, Job, Level, Permissions, Push, Run, Step, Strategy, Use,
    Workflow, WorkflowDispatch, WorkflowRun, WorkflowRunType,
};
use indoc::formatdoc;
use serde_json::{Value, json};

use crate::product_manifest::ProductManifest;

use crate::tasks::workflows::{
    run_bundling::upload_artifact,
    runners,
    steps::{
        self, CommonPermissionSets, DownloadArtifactStep, IfNoFilesFound, NamedJob, dependant_job,
        named,
    },
    vars::{self, JobOutput, StepOutput, WorkflowInput, assets},
};

const CURRENT_ACTION_RUN_URL: &str =
    "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}";

pub(crate) fn release() -> Workflow {
    // GitHub marks removed workflows as deleted and will not disable them. Keep
    // this path so its historical identity can remain explicitly disabled.
    named::workflow()
        .on(Event::default().workflow_dispatch(WorkflowDispatch::default()))
        .with_minimal_permissions()
        .add_job(
            "retired",
            Job::default()
                .cond(Expression::new("false"))
                .runs_on("ubuntu-22.04")
                .add_step(named::bash("exit 1")),
        )
}

pub(crate) fn release_promotion() -> Workflow {
    let bump = WorkflowInput::string("bump", Some("patch".to_owned()))
        .description("Version component for a manual release: patch, minor, or major");
    let version = WorkflowInput::string("version", Some(String::new()))
        .description("Optional exact X.Y.Z version for manual recovery");
    let commit_sha = WorkflowInput::string("commit_sha", Some(String::new()))
        .description("Optional tested release commit SHA; defaults to the selected release commit");
    let dispatch = WorkflowDispatch::default()
        .add_input(bump.name, bump.input())
        .add_input(version.name, version.input())
        .add_input(commit_sha.name, commit_sha.input());
    let (prepare, prepared) = prepare_release();
    let preflight = signing_preflight(&prepare, &prepared);
    let licenses = release_licenses(&prepare, &prepared);
    let applications = application_builds(&prepare, &prepared, &licenses);
    let remote_servers = remote_server_builds(&prepare, &prepared);
    let build = product_builds(
        &prepare,
        &prepared,
        &licenses,
        &applications,
        &remote_servers,
    );
    let publish = publish_product(&prepare, &prepared, &build);
    let refresh = refresh_website(&publish, &prepare, &prepared);

    named::workflow()
        .name("release_promotion")
        .on(Event::default().workflow_dispatch(dispatch).workflow_run(
            WorkflowRun::default()
                .add_workflow("run_tests")
                .add_type(WorkflowRunType::Completed)
                .add_branch("release"),
        ))
        .concurrency(Concurrency::new(Expression::new("copper-release")).cancel_in_progress(false))
        .with_minimal_permissions()
        .add_env(("CARGO_TERM_COLOR", "always"))
        .add_env(("RUST_BACKTRACE", "1"))
        .add_job(prepare.name, prepare.job)
        .add_job(preflight.name, preflight.job)
        .add_job(licenses.name, licenses.job)
        .add_job(applications.name, applications.job)
        .add_job(remote_servers.name, remote_servers.job)
        .add_job(build.name, build.job)
        .add_job(publish.name, publish.job)
        .add_job(refresh.name, refresh.job)
}

pub(crate) fn release_signing_check() -> Workflow {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let initial_version = manifest
        .product("rust")
        .expect("Copper product")
        .initial_version
        .clone()
        .expect("Copper initial version");
    let version = WorkflowInput::string("version", Some(initial_version.clone()))
        .description("Exact package version to validate; no tag or release is created");
    let sign_artifacts = WorkflowInput::bool("sign_artifacts", Some(false)).description(
        "Contact Apple/Azure and build signed packages; requires approval and production credentials",
    );
    let resolve = named::bash(indoc::indoc! {r#"
        python3 - <<'PYVERSION'
        import os
        import re
        if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", os.environ["VERSION"]):
            raise SystemExit("version must be exactly X.Y.Z")
        PYVERSION
        {
            echo "commit_sha=$COMMIT_SHA"
            echo "version=$VERSION"
            echo "tag="
        } >> "$GITHUB_OUTPUT"
    "#})
    .id("resolve-signing-check")
    .add_env(("VERSION", format!("${{{{ inputs.version || '{initial_version}' }}}}")))
    .add_env(("COMMIT_SHA", "${{ github.sha }}"));
    let commit_sha = StepOutput::new(&resolve, "commit_sha");
    let package_version = StepOutput::new(&resolve, "version");
    let tag = StepOutput::new(&resolve, "tag");
    let prepare = NamedJob {
        name: "prepare_release".to_owned(),
        job: Job::default()
            .runs_on("ubuntu-22.04")
            .timeout_minutes(20u32)
            .permissions(Permissions::default().contents(Level::Read))
            .add_step(steps::checkout_repo().with_ref("${{ github.sha }}"))
            .add_step(setup_signing_python())
            .add_step(resolve)
            .add_step(named::bash(
                "CARGO_BUILD_JOBS=2 cargo build -p xtask --locked",
            ))
            .add_step(named::bash(
                "python3 script/check-repository-inputs && python3 script/test-production-signing && cargo test -p xtask --locked",
            ))
            .add_step(
                generate_release_bundle_plans()
                    .add_env(("COMMIT_SHA", commit_sha.to_string()))
                    .add_env(("RELEASE_VERSION", package_version.to_string())),
            )
            .add_step(
                steps::upload_artifact("release-bundle-plans", "release-bundle-plans")
                    .if_no_files_found(IfNoFilesFound::Error),
            )
            .outputs([
                (commit_sha.name.to_owned(), commit_sha.to_string()),
                (package_version.name.to_owned(), package_version.to_string()),
                (tag.name.to_owned(), tag.to_string()),
            ]),
    };
    let prepared = PreparedRelease {
        commit_sha: commit_sha.as_job_output(&prepare),
        version: package_version.as_job_output(&prepare),
        tag: tag.as_job_output(&prepare),
    };
    let mut preflight = signing_preflight(&prepare, &prepared);
    preflight.job = preflight.job.cond(Expression::new(
        "github.event_name == 'workflow_dispatch' && inputs.sign_artifacts == true",
    ));
    let licenses = release_licenses(&prepare, &prepared);
    let applications = application_builds(&prepare, &prepared, &licenses);
    let remote_servers = remote_server_builds(&prepare, &prepared);
    let packages = product_builds(
        &prepare,
        &prepared,
        &licenses,
        &applications,
        &remote_servers,
    );
    let contract_push = Push::default()
        .add_branch("codex/**")
        .add_branch("main")
        .add_path(".github/workflows/release_signing_check.yml")
        .add_path("tooling/xtask/src/tasks/workflows/release.rs")
        .add_path("script/check-signing-environment")
        .add_path("script/test-production-signing")
        .add_path("script/sign-linux-archive")
        .add_path("script/lib/macos-signing.sh");
    named::workflow()
        .name("release_signing_check")
        .on(Event::default().push(contract_push).workflow_dispatch(
            WorkflowDispatch::default()
                .add_input(version.name, version.input())
                .add_input(sign_artifacts.name, sign_artifacts.input()),
        ))
        .concurrency(
            Concurrency::new(Expression::new("copper-signing-check")).cancel_in_progress(false),
        )
        .with_minimal_permissions()
        .add_job(prepare.name, prepare.job)
        .add_job(preflight.name, preflight.job)
        .add_job(licenses.name, licenses.job)
        .add_job(applications.name, applications.job)
        .add_job(remote_servers.name, remote_servers.job)
        .add_job(packages.name, packages.job)
}

fn setup_signing_python() -> Step<Use> {
    named::uses(
        "actions",
        "setup-python",
        "a26af69be951a213d495a4c3e4e4022e16d87065",
    )
    .add_with(("python-version", "3.12"))
}

fn signing_preflight(prepare: &NamedJob, prepared: &PreparedRelease) -> NamedJob {
    let mut job = Job::default()
        .needs([prepare.name.clone()])
        .runs_on("ubuntu-22.04")
        .timeout_minutes(5u32)
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(steps::checkout_repo().with_ref(prepared.commit_sha.to_string()));
    for (platform, names) in [
        (
            "macos",
            &[
                "MACOS_SIGNING_IDENTITY",
                "MACOS_CERTIFICATE",
                "MACOS_CERTIFICATE_PASSWORD",
                "APPLE_NOTARIZATION_KEY",
                "APPLE_NOTARIZATION_KEY_ID",
                "APPLE_NOTARIZATION_ISSUER_ID",
            ][..],
        ),
        (
            "windows",
            &[
                "AZURE_TENANT_ID",
                "AZURE_CLIENT_ID",
                "AZURE_CLIENT_SECRET",
                "ACCOUNT_NAME",
                "CERT_PROFILE_NAME",
                "ENDPOINT",
                "WINDOWS_SIGNING_PUBLISHER",
                "FILE_DIGEST",
                "TIMESTAMP_DIGEST",
                "TIMESTAMP_SERVER",
            ][..],
        ),
        (
            "linux",
            &[
                "LINUX_SIGNING_KEY",
                "LINUX_SIGNING_KEY_PASSWORD",
                "LINUX_SIGNING_KEY_FINGERPRINT",
            ][..],
        ),
    ] {
        let mut step = named::bash(format!(
            "python3 script/check-signing-environment --platform {platform} --policy required"
        ));
        for name in names {
            step = step.add_env((*name, format!("${{{{ secrets.{name} }}}}")));
        }
        job = job.add_step(step);
    }
    named::job(job)
}

const RELEASE_COMMIT: &str = "${{ github.event_name == 'workflow_run' && github.event.workflow_run.head_sha || (github.event_name == 'workflow_dispatch' && inputs.commit_sha || github.sha) }}";
const RELEASE_BUMP: &str =
    "${{ github.event_name == 'workflow_dispatch' && inputs.bump || 'patch' }}";
const RELEASE_EXPLICIT_VERSION: &str =
    "${{ github.event_name == 'workflow_dispatch' && inputs.version || '' }}";
const AUTOMATIC_RELEASE_GUARD: &str = "vars.COPPER_RELEASE_ENABLED == 'true' && vars.COPPER_RELEASE_BOOTSTRAP_SHA != '' && ((github.event_name == 'workflow_run' && github.event.workflow_run.conclusion == 'success' && github.event.workflow_run.event == 'push' && github.event.workflow_run.head_branch == 'release' && github.event.workflow_run.head_repository.full_name == github.repository && github.event.workflow_run.head_sha != vars.COPPER_RELEASE_BOOTSTRAP_SHA) || (github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/release'))";

fn release_promotion_guard() -> &'static str {
    indoc::indoc! {r#"
        set -euo pipefail
        if [[ "$RELEASE_ENABLED" != "true" || ! "$BOOTSTRAP_SHA" =~ ^[0-9a-f]{40}$ || ! "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
            echo "Release promotion is disabled or the bootstrap/commit SHA is invalid" >&2
            exit 1
        fi
        git fetch origin +refs/heads/release:refs/remotes/origin/release --tags
        if [[ "$COMMIT_SHA" == "$BOOTSTRAP_SHA" ]]; then
            echo "The initial release-branch commit can never be released" >&2
            exit 1
        fi
        git merge-base --is-ancestor "$BOOTSTRAP_SHA" "$COMMIT_SHA"
        RELEASE_HISTORY="$(git rev-list --first-parent origin/release)"
        if ! grep -Fxq "$COMMIT_SHA" <<<"$RELEASE_HISTORY"; then
            echo "Release commit is not on the first-parent history of origin/release" >&2
            exit 1
        fi
        CI_RUNS="$(gh api --method GET "repos/$GITHUB_REPOSITORY/actions/workflows/run_tests.yml/runs" \
            -f branch=release -f event=push -f head_sha="$COMMIT_SHA" -f per_page=100 --paginate --slurp)"
        if ! jq -e --arg commit "$COMMIT_SHA" --arg repository "$GITHUB_REPOSITORY" --arg run_id "$CI_RUN_ID" '
            [.[].workflow_runs[]] | max_by(.id) |
            .head_sha == $commit and .head_branch == "release" and .event == "push" and
            .head_repository.full_name == $repository and .repository.full_name == $repository and
            .path == ".github/workflows/run_tests.yml" and
            .status == "completed" and .conclusion == "success" and
            ($run_id == "" or (.id | tostring) == $run_id)
        ' <<<"$CI_RUNS" >/dev/null; then
            echo "The latest exact-commit release push CI must have completed successfully in this repository" >&2
            exit 1
        fi
    "#}
}

fn promotion_check() -> Step<Run> {
    named::bash(release_promotion_guard())
        .add_env(("COMMIT_SHA", RELEASE_COMMIT))
        .add_env(("BOOTSTRAP_SHA", "${{ vars.COPPER_RELEASE_BOOTSTRAP_SHA }}"))
        .add_env(("RELEASE_ENABLED", "${{ vars.COPPER_RELEASE_ENABLED }}"))
        .add_env((
            "CI_RUN_ID",
            "${{ github.event_name == 'workflow_run' && github.event.workflow_run.id || '' }}",
        ))
        .add_env(("GH_TOKEN", vars::GITHUB_TOKEN))
}

struct PreparedRelease {
    commit_sha: JobOutput,
    version: JobOutput,
    tag: JobOutput,
}

fn prepare_release() -> (NamedJob, PreparedRelease) {
    let resolve = named::bash(indoc::indoc! {r#"
        git fetch origin +refs/heads/release:refs/remotes/origin/release --tags

        ARGS=(release-version --commit "$COMMIT_SHA" --bootstrap-commit "$BOOTSTRAP_SHA" --bump "$BUMP")
        if [[ -n "$EXPLICIT_VERSION" ]]; then
            ARGS+=(--explicit-version "$EXPLICIT_VERSION")
        fi
        DECISION="$(cargo xtask "${ARGS[@]}")"
        jq -e '.commit_sha and .version and .tag' <<<"$DECISION" >/dev/null
        {
            echo "commit_sha=$(jq -r '.commit_sha' <<<"$DECISION")"
            echo "version=$(jq -r '.version' <<<"$DECISION")"
            echo "tag=$(jq -r '.tag' <<<"$DECISION")"
        } >> "$GITHUB_OUTPUT"
    "#})
    .id("resolve-release")
    .add_env(("COMMIT_SHA", RELEASE_COMMIT))
    .add_env(("BUMP", RELEASE_BUMP))
    .add_env(("EXPLICIT_VERSION", RELEASE_EXPLICIT_VERSION))
    .add_env(("BOOTSTRAP_SHA", "${{ vars.COPPER_RELEASE_BOOTSTRAP_SHA }}"));
    let commit_sha = StepOutput::new(&resolve, "commit_sha");
    let version = StepOutput::new(&resolve, "version");
    let tag = StepOutput::new(&resolve, "tag");
    let generate_bundle_plans = generate_release_bundle_plans()
        .add_env(("COMMIT_SHA", commit_sha.to_string()))
        .add_env(("RELEASE_VERSION", version.to_string()));
    let job = named::job(
        Job::default()
            .cond(Expression::new(AUTOMATIC_RELEASE_GUARD))
            .runs_on("ubuntu-22.04")
            .timeout_minutes(15u32)
            .permissions(
                Permissions::default()
                    .contents(Level::Read)
                    .actions(Level::Read),
            )
            .add_step(steps::checkout_repo().with_ref("main").with_full_history())
            .add_step(promotion_check())
            .add_step(
                steps::checkout_repo()
                    .with_ref(RELEASE_COMMIT)
                    .with_full_history()
                    .with_fetch_tags(),
            )
            .add_step(resolve)
            .add_step(generate_bundle_plans)
            .add_step(
                steps::upload_artifact("release-bundle-plans", "release-bundle-plans")
                    .if_no_files_found(IfNoFilesFound::Error),
            )
            .outputs([
                (commit_sha.name.to_owned(), commit_sha.to_string()),
                (version.name.to_owned(), version.to_string()),
                (tag.name.to_owned(), tag.to_string()),
            ]),
    );
    let outputs = PreparedRelease {
        commit_sha: commit_sha.as_job_output(&job),
        version: version.as_job_output(&job),
        tag: tag.as_job_output(&job),
    };
    (job, outputs)
}

fn product_matrix() -> Vec<Value> {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let mut include = Vec::new();
    for product in manifest.enabled_products() {
        let application_features = product.cargo_features.join(",");
        let remote_features = product.remote_server_features.join(",");
        for target in &product.targets {
            let row = match target.as_str() {
                "linux-x86_64" => json!({
                    "product": product.id,
                    "display_name": product.display_name,
                    "executable_name": product.executable_name,
                    "bundle_identifier": product.bundle_identifier,
                    "url_scheme": product.url_scheme,
                    "icon_set": product.icon_set,
                    "data_namespace": product.data_namespace,
                    "update_namespace": product.update_namespace,
                    "windows_installer_id": product.windows_installer_id,
                    "platform": "linux",
                    "runner": "ubuntu-22.04",
                    "target": "x86_64-unknown-linux-gnu",
                    "remote_target": "x86_64-unknown-linux-musl",
                    "application_features": application_features,
                    "remote_features": remote_features,
                    "application_artifact": format!("{}-application-linux-x86_64", product.id),
                    "remote_artifact": format!("{}-remote-server-linux-x86_64", product.id),
                    "plan_file": format!("{}-linux-x86_64.json", product.id),
                    "artifact": format!("{}-linux-x86_64", product.id),
                    "artifact_suffix": "linux-x86_64.tar.gz",
                    "artifact_path": format!("target/products/{}/release/{}-*-linux-x86_64.tar.gz*", product.id, product.id),
                }),
                "macos-aarch64" => json!({
                    "product": product.id,
                    "display_name": product.display_name,
                    "executable_name": product.executable_name,
                    "bundle_identifier": product.bundle_identifier,
                    "url_scheme": product.url_scheme,
                    "icon_set": product.icon_set,
                    "data_namespace": product.data_namespace,
                    "update_namespace": product.update_namespace,
                    "windows_installer_id": product.windows_installer_id,
                    "platform": "macos",
                    "runner": "macos-15",
                    "target": "aarch64-apple-darwin",
                    "remote_target": "aarch64-apple-darwin",
                    "application_features": application_features,
                    "remote_features": remote_features,
                    "application_artifact": format!("{}-application-macos-aarch64", product.id),
                    "remote_artifact": format!("{}-remote-server-macos-aarch64", product.id),
                    "plan_file": format!("{}-macos-aarch64.json", product.id),
                    "artifact": format!("{}-macos-aarch64", product.id),
                    "artifact_suffix": "macos-aarch64.dmg",
                    "artifact_path": format!("target/products/{}/release/{}-*-macos-aarch64.dmg", product.id, product.id),
                }),
                "windows-x86_64" => json!({
                    "product": product.id,
                    "display_name": product.display_name,
                    "executable_name": product.executable_name,
                    "bundle_identifier": product.bundle_identifier,
                    "url_scheme": product.url_scheme,
                    "icon_set": product.icon_set,
                    "data_namespace": product.data_namespace,
                    "update_namespace": product.update_namespace,
                    "windows_installer_id": product.windows_installer_id,
                    "platform": "windows",
                    "runner": "windows-2022",
                    "target": "x86_64-pc-windows-msvc",
                    "remote_target": "x86_64-pc-windows-msvc",
                    "application_features": application_features,
                    "remote_features": remote_features,
                    "application_artifact": format!("{}-application-windows-x86_64", product.id),
                    "remote_artifact": format!("{}-remote-server-windows-x86_64", product.id),
                    "plan_file": format!("{}-windows-x86_64.json", product.id),
                    "artifact": format!("{}-windows-x86_64", product.id),
                    "artifact_suffix": "windows-x86_64.exe",
                    "artifact_path": format!("target/products/{}/release/{}-*-windows-x86_64.exe", product.id, product.id),
                }),
                unsupported => {
                    panic!("validated catalog contains unsupported target {unsupported}")
                }
            };
            include.push(row);
        }
    }
    include
}

fn generate_release_bundle_plans() -> Step<Run> {
    let mut command = String::from("mkdir -p release-bundle-plans\n");
    for row in product_matrix() {
        let product = row["product"].as_str().expect("product is a string");
        let platform = row["platform"].as_str().expect("platform is a string");
        let target = row["target"].as_str().expect("target is a string");
        let plan_file = row["plan_file"].as_str().expect("plan file is a string");
        command.push_str(&format!(
            "target/debug/xtask bundle --product {product} --platform {platform} --target {target} --signing required --dry-run \\\n    | jq --arg commit_sha \"$COMMIT_SHA\" '. + {{commit_sha: $commit_sha}}' > release-bundle-plans/{plan_file}\n"
        ));
    }
    named::bash(command)
}

fn release_cache_environment(job: Job) -> Job {
    job.add_env(("SCCACHE_GHA_ENABLED", "on"))
        .add_env(("SCCACHE_IDLE_TIMEOUT", "0"))
        .add_env(("RUSTC_WRAPPER", "sccache"))
}

fn release_bundle_environment(job: Job) -> Job {
    job.add_env(("CARGO_TARGET_DIR", "target/products/${{ matrix.product }}"))
        .add_env(("ZED_PRODUCT_ID", "${{ matrix.product }}"))
        .add_env(("ZED_PRODUCT_DISPLAY_NAME", "${{ matrix.display_name }}"))
        .add_env(("ZED_PRODUCT_EXECUTABLE", "${{ matrix.executable_name }}"))
        .add_env(("ZED_PRODUCT_BUNDLE_ID", "${{ matrix.bundle_identifier }}"))
        .add_env(("ZED_PRODUCT_URL_SCHEME", "${{ matrix.url_scheme }}"))
        .add_env((
            "ZED_PRODUCT_ICON_SET",
            "${{ github.workspace }}/${{ matrix.icon_set }}",
        ))
        .add_env(("ZED_PRODUCT_DATA_NAMESPACE", "${{ matrix.data_namespace }}"))
        .add_env((
            "ZED_PRODUCT_UPDATE_NAMESPACE",
            "${{ matrix.update_namespace }}",
        ))
        .add_env((
            "ZED_PRODUCT_WINDOWS_INSTALLER_ID",
            "${{ matrix.windows_installer_id }}",
        ))
        .add_env((
            "ZED_PRODUCT_APP_FEATURES",
            "${{ matrix.application_features }}",
        ))
        .add_env((
            "ZED_PRODUCT_REMOTE_FEATURES",
            "${{ matrix.remote_features }}",
        ))
        .add_env(("ZED_RELEASE_CHANNEL", "stable"))
}

fn cache_macos_cargo_bundle() -> Step<Use> {
    named::uses(
        "actions",
        "cache",
        "0057852bfaa89a56745cba8c7296529d2fc39830", // v4.3.0
    )
    .add_with(("path", "~/.cargo/bin/cargo-bundle"))
    .add_with((
        "key",
        "release-tool-v1-cargo-bundle-0.11.0-${{ runner.os }}-${{ runner.arch }}",
    ))
}

fn configure_release_cache_namespace(target: &str) -> Step<Run> {
    named::bash(indoc::indoc! {r#"
        TOOLCHAIN_HASH="$(git hash-object rust-toolchain.toml)"
        echo "SCCACHE_GHA_VERSION=release-v2-${RUNNER_OS}-${CACHE_TARGET}-${TOOLCHAIN_HASH}" >> "$GITHUB_ENV"
    "#})
    .add_env(("CACHE_TARGET", target))
}

fn release_licenses(prepare: &NamedJob, prepared: &PreparedRelease) -> NamedJob {
    let stage = named::bash(indoc::indoc! {r#"
        sha256_file() {
            sha256sum "$1" | awk '{print $1}'
        }
        test "$(git rev-parse HEAD)" = "$COMMIT_SHA"
        git diff --quiet "$COMMIT_SHA" -- Cargo.lock
        script/generate-licenses
        mkdir -p release-licenses
        cp assets/licenses.md release-licenses/licenses.md
        CARGO_LOCK_BLOB="$(git rev-parse "${COMMIT_SHA}:Cargo.lock")"
        LICENSES_HASH="$(sha256_file release-licenses/licenses.md)"
        jq -n \
            --arg commit_sha "$COMMIT_SHA" \
            --arg cargo_lock_blob "$CARGO_LOCK_BLOB" \
            --arg licenses_hash "$LICENSES_HASH" \
            '{commit_sha: $commit_sha, cargo_lock_blob: $cargo_lock_blob, licenses_hash: $licenses_hash}' \
            > release-licenses/metadata.json
    "#})
    .add_env(("COMMIT_SHA", prepared.commit_sha.to_string()));

    named::job(
        Job::default()
            .needs([prepare.name.clone(), "signing_preflight".to_owned()])
            .runs_on("ubuntu-22.04")
            .timeout_minutes(20u32)
            .permissions(Permissions::default().contents(Level::Read))
            .add_step(steps::checkout_repo().with_ref(prepared.commit_sha.to_string()))
            .add_step(steps::install_cargo_about())
            .add_step(stage)
            .add_step(
                steps::upload_artifact("release-licenses", "release-licenses")
                    .if_no_files_found(IfNoFilesFound::Error),
            ),
    )
}

fn verify_release_licenses() -> Step<Run> {
    named::bash(indoc::indoc! {r#"
        sha256_file() {
            if [[ "$RUNNER_OS" == "macOS" ]]; then
                shasum -a 256 "$1" | awk '{print $1}'
            else
                sha256sum "$1" | awk '{print $1}'
            fi
        }
        test -s release-licenses/licenses.md
        test -s release-licenses/metadata.json
        test "$(git rev-parse HEAD)" = "$COMMIT_SHA"
        git diff --quiet "$COMMIT_SHA" -- Cargo.lock
        CARGO_LOCK_BLOB="$(git rev-parse "${COMMIT_SHA}:Cargo.lock")"
        LICENSES_HASH="$(sha256_file release-licenses/licenses.md)"
        jq -e \
            --arg commit_sha "$COMMIT_SHA" \
            --arg cargo_lock_blob "$CARGO_LOCK_BLOB" \
            --arg licenses_hash "$LICENSES_HASH" \
            '.commit_sha == $commit_sha and .cargo_lock_blob == $cargo_lock_blob and .licenses_hash == $licenses_hash' \
            release-licenses/metadata.json >/dev/null
        cp release-licenses/licenses.md assets/licenses.md
    "#})
    .add_env(("COMMIT_SHA", "${{ needs.prepare_release.outputs.commit_sha }}"))
}

fn application_builds(
    prepare: &NamedJob,
    prepared: &PreparedRelease,
    licenses: &NamedJob,
) -> NamedJob {
    let job = Job::default()
        .needs([prepare.name.clone(), licenses.name.clone()])
        .runs_on("${{ matrix.runner }}")
        .timeout_minutes(180u32)
        .permissions(Permissions::default().contents(Level::Read))
        .add_env(("RELEASE_VERSION", prepared.version.to_string()))
        .strategy(
            Strategy::default()
                .fail_fast(false)
                .matrix(json!({ "include": product_matrix() })),
        )
        .add_step(steps::checkout_repo().with_ref(prepared.commit_sha.to_string()))
        .add_step(
            steps::download_artifact()
                .artifact_name("release-licenses")
                .path("release-licenses"),
        )
        .add_step(verify_release_licenses())
        .add_step(configure_release_cache_namespace("${{ matrix.target }}"))
        .add_step(steps::setup_release_sccache())
        .add_step(
            steps::enable_windows_long_paths()
                .if_condition(Expression::new("matrix.platform == 'windows'")),
        )
        .add_step(
            named::bash("./script/linux && ./script/download-wasi-sdk")
                .if_condition(Expression::new("matrix.platform == 'linux'")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                export ZED_PRODUCT_ARTIFACT_NAME="${ZED_PRODUCT_ID}-${RELEASE_VERSION}-${ARTIFACT_SUFFIX}"
                export ZED_BUNDLE_PHASE=application
                script/bundle-linux
            "#})
                .if_condition(Expression::new("matrix.platform == 'linux'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("ZED_PREGENERATED_LICENSES", "1"))
                .add_env(("CC", "clang"))
                .add_env(("CXX", "clang++")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                export ZED_PRODUCT_ARTIFACT_NAME="${ZED_PRODUCT_ID}-${RELEASE_VERSION}-${ARTIFACT_SUFFIX}"
                export ZED_BUNDLE_PHASE=application
                script/bundle-mac "$TARGET"
            "#})
                .if_condition(Expression::new("matrix.platform == 'macos'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("ZED_PREGENERATED_LICENSES", "1")),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                cargo test -p auto_update_helper --locked
                if ($LASTEXITCODE -ne 0) { throw "Windows updater tests failed" }
                bash ./script/clippy --no-all-features -p auto_update_helper
                if ($LASTEXITCODE -ne 0) { throw "Windows updater Clippy failed" }
            "#})
                .if_condition(Expression::new("matrix.platform == 'windows'")),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $env:ZED_PRODUCT_ARTIFACT_NAME = "$env:ZED_PRODUCT_ID-$env:RELEASE_VERSION-$env:ARTIFACT_SUFFIX"
                $env:ZED_BUNDLE_PHASE = "application"
                script/bundle-windows.ps1 -Architecture x86_64
            "#})
                .if_condition(Expression::new("matrix.platform == 'windows'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("ZED_PREGENERATED_LICENSES", "1")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                SOURCE="target/products/${PRODUCT_ID}/${TARGET}/release"
                mkdir -p release-application
                cp "$SOURCE/zed" "$SOURCE/cli" release-application/
                if [[ "$TARGET" == *-apple-darwin ]]; then
                    cp "$SOURCE/native-cargo-runtime-observer" release-application/
                fi
                jq -n --arg commit_sha "$COMMIT_SHA" --arg target "$TARGET" --arg features "$FEATURES" \
                    '{commit_sha: $commit_sha, target: $target, features: $features}' > release-application/metadata.json
            "#})
            .if_condition(Expression::new("matrix.platform != 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("TARGET", "${{ matrix.target }}"))
            .add_env(("FEATURES", "${{ matrix.application_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $source = "target/products/$env:PRODUCT_ID/$env:TARGET/release"
                New-Item -ItemType Directory -Path release-application -Force | Out-Null
                Copy-Item "$source/zed.exe", "$source/cli.exe", "$source/auto_update_helper.exe", "$source/zed.pdb", "$source/cli.pdb", "$source/auto_update_helper.pdb" -Destination release-application
                @{
                    commit_sha = $env:COMMIT_SHA
                    target = $env:TARGET
                    features = $env:FEATURES
                } | ConvertTo-Json | Set-Content release-application/metadata.json
            "#})
            .if_condition(Expression::new("matrix.platform == 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("TARGET", "${{ matrix.target }}"))
            .add_env(("FEATURES", "${{ matrix.application_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            steps::upload_artifact("${{ matrix.application_artifact }}", "release-application")
                .if_no_files_found(IfNoFilesFound::Error),
        )
        .add_step(
            steps::finalize_release_sccache(runners::Platform::Linux)
                .if_condition(Expression::new("always() && matrix.platform != 'windows'")),
        )
        .add_step(
            steps::finalize_release_sccache(runners::Platform::Windows)
                .if_condition(Expression::new("always() && matrix.platform == 'windows'")),
        );
    named::job(release_bundle_environment(release_cache_environment(job)))
}

fn remote_server_builds(prepare: &NamedJob, prepared: &PreparedRelease) -> NamedJob {
    let job = Job::default()
        .needs([prepare.name.clone(), "signing_preflight".to_owned()])
        .runs_on("${{ matrix.runner }}")
        .timeout_minutes(120u32)
        .permissions(Permissions::default().contents(Level::Read))
        .add_env(("RELEASE_VERSION", prepared.version.to_string()))
        .strategy(
            Strategy::default()
                .fail_fast(false)
                .matrix(json!({ "include": product_matrix() })),
        )
        .add_step(steps::checkout_repo().with_ref(prepared.commit_sha.to_string()))
        .add_step(configure_release_cache_namespace(
            "${{ matrix.remote_target }}",
        ))
        .add_step(steps::setup_release_sccache())
        .add_step(
            steps::enable_windows_long_paths()
                .if_condition(Expression::new("matrix.platform == 'windows'")),
        )
        .add_step(
            named::bash("./script/linux && ./script/download-wasi-sdk")
                .if_condition(Expression::new("matrix.platform == 'linux'")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                export ZED_PRODUCT_ARTIFACT_NAME="${ZED_PRODUCT_ID}-${RELEASE_VERSION}-${ARTIFACT_SUFFIX}"
                export ZED_BUNDLE_PHASE=remote-server
                script/bundle-linux
            "#})
                .if_condition(Expression::new("matrix.platform == 'linux'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("CC", "clang"))
                .add_env(("CXX", "clang++")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                export ZED_PRODUCT_ARTIFACT_NAME="${ZED_PRODUCT_ID}-${RELEASE_VERSION}-${ARTIFACT_SUFFIX}"
                export ZED_BUNDLE_PHASE=remote-server
                script/bundle-mac "$TARGET"
            "#})
                .if_condition(Expression::new("matrix.platform == 'macos'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("TARGET", "${{ matrix.target }}")),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $env:ZED_PRODUCT_ARTIFACT_NAME = "$env:ZED_PRODUCT_ID-$env:RELEASE_VERSION-$env:ARTIFACT_SUFFIX"
                $env:ZED_BUNDLE_PHASE = "remote-server"
                script/bundle-windows.ps1 -Architecture x86_64
            "#})
                .if_condition(Expression::new("matrix.platform == 'windows'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                SOURCE="target/products/${PRODUCT_ID}/${REMOTE_TARGET}/release"
                mkdir -p release-remote-server
                cp "$SOURCE/remote_server" release-remote-server/
                jq -n --arg commit_sha "$COMMIT_SHA" --arg target "$REMOTE_TARGET" --arg features "$FEATURES" \
                    '{commit_sha: $commit_sha, target: $target, features: $features}' > release-remote-server/metadata.json
            "#})
            .if_condition(Expression::new("matrix.platform != 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("REMOTE_TARGET", "${{ matrix.remote_target }}"))
            .add_env(("FEATURES", "${{ matrix.remote_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $source = "target/products/$env:PRODUCT_ID/$env:REMOTE_TARGET/release"
                New-Item -ItemType Directory -Path release-remote-server -Force | Out-Null
                Copy-Item "$source/remote_server.exe", "$source/remote_server.pdb" -Destination release-remote-server
                @{
                    commit_sha = $env:COMMIT_SHA
                    target = $env:REMOTE_TARGET
                    features = $env:FEATURES
                } | ConvertTo-Json | Set-Content release-remote-server/metadata.json
            "#})
            .if_condition(Expression::new("matrix.platform == 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("REMOTE_TARGET", "${{ matrix.remote_target }}"))
            .add_env(("FEATURES", "${{ matrix.remote_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            steps::upload_artifact("${{ matrix.remote_artifact }}", "release-remote-server")
                .if_no_files_found(IfNoFilesFound::Error),
        )
        .add_step(
            steps::finalize_release_sccache(runners::Platform::Linux)
                .if_condition(Expression::new("always() && matrix.platform != 'windows'")),
        )
        .add_step(
            steps::finalize_release_sccache(runners::Platform::Windows)
                .if_condition(Expression::new("always() && matrix.platform == 'windows'")),
        );
    named::job(release_bundle_environment(release_cache_environment(job)))
}

fn product_builds(
    prepare: &NamedJob,
    prepared: &PreparedRelease,
    licenses: &NamedJob,
    applications: &NamedJob,
    remote_servers: &NamedJob,
) -> NamedJob {
    let job = Job::default()
        .needs([
            prepare.name.clone(),
            licenses.name.clone(),
            applications.name.clone(),
            remote_servers.name.clone(),
        ])
        .runs_on("${{ matrix.runner }}")
        .timeout_minutes(90u32)
        .permissions(Permissions::default().contents(Level::Read))
        .add_env(("RELEASE_VERSION", prepared.version.to_string()))
        .add_env(("ZED_PRODUCT_SIGNING", "required"))
        .strategy(
            Strategy::default()
                .fail_fast(false)
                .matrix(json!({ "include": product_matrix() })),
        )
        .add_step(steps::checkout_repo().with_ref(prepared.commit_sha.to_string()))
        .add_step(setup_signing_python())
        .add_step(
            steps::download_artifact()
                .artifact_name("release-licenses")
                .path("release-licenses"),
        )
        .add_step(
            steps::download_artifact()
                .artifact_name("release-bundle-plans")
                .path("release-bundle-plans"),
        )
        .add_step(
            steps::download_artifact()
                .artifact_name("${{ matrix.application_artifact }}")
                .path("release-application"),
        )
        .add_step(
            steps::download_artifact()
                .artifact_name("${{ matrix.remote_artifact }}")
                .path("release-remote-server"),
        )
        .add_step(verify_release_licenses())
        .add_step(
            named::bash(indoc::indoc! {r#"
                jq -e --arg commit_sha "$COMMIT_SHA" --arg target "$TARGET" --arg features "$APPLICATION_FEATURES" \
                    '.commit_sha == $commit_sha and .target == $target and .features == $features' \
                    release-application/metadata.json >/dev/null
                jq -e --arg commit_sha "$COMMIT_SHA" --arg target "$REMOTE_TARGET" --arg features "$REMOTE_FEATURES" \
                    '.commit_sha == $commit_sha and .target == $target and .features == $features' \
                    release-remote-server/metadata.json >/dev/null
                APPLICATION_DEST="target/products/${PRODUCT_ID}/${TARGET}/release"
                REMOTE_DEST="target/products/${PRODUCT_ID}/${REMOTE_TARGET}/release"
                mkdir -p "$APPLICATION_DEST" "$REMOTE_DEST"
                cp release-application/zed release-application/cli "$APPLICATION_DEST/"
                cp release-remote-server/remote_server "$REMOTE_DEST/"
                if [[ "$TARGET" == *-apple-darwin ]]; then
                    cp release-application/native-cargo-runtime-observer "$APPLICATION_DEST/"
                    chmod +x "$APPLICATION_DEST/native-cargo-runtime-observer"
                fi
                chmod +x "$APPLICATION_DEST/zed" "$APPLICATION_DEST/cli" "$REMOTE_DEST/remote_server"
            "#})
            .if_condition(Expression::new("matrix.platform != 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("TARGET", "${{ matrix.target }}"))
            .add_env(("REMOTE_TARGET", "${{ matrix.remote_target }}"))
            .add_env(("APPLICATION_FEATURES", "${{ matrix.application_features }}"))
            .add_env(("REMOTE_FEATURES", "${{ matrix.remote_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $applicationMetadata = Get-Content release-application/metadata.json | ConvertFrom-Json
                $remoteMetadata = Get-Content release-remote-server/metadata.json | ConvertFrom-Json
                if ($applicationMetadata.commit_sha -ne $env:COMMIT_SHA -or $applicationMetadata.target -ne $env:TARGET -or $applicationMetadata.features -ne $env:APPLICATION_FEATURES) {
                    throw "Application artifact metadata does not match the release plan"
                }
                if ($remoteMetadata.commit_sha -ne $env:COMMIT_SHA -or $remoteMetadata.target -ne $env:REMOTE_TARGET -or $remoteMetadata.features -ne $env:REMOTE_FEATURES) {
                    throw "Remote-server artifact metadata does not match the release plan"
                }
                $applicationDest = "target/products/$env:PRODUCT_ID/$env:TARGET/release"
                $remoteDest = "target/products/$env:PRODUCT_ID/$env:REMOTE_TARGET/release"
                New-Item -ItemType Directory -Path $applicationDest, $remoteDest -Force | Out-Null
                Copy-Item release-application/zed.exe, release-application/cli.exe, release-application/auto_update_helper.exe, release-application/zed.pdb, release-application/cli.pdb, release-application/auto_update_helper.pdb -Destination $applicationDest
                Copy-Item release-remote-server/remote_server.exe, release-remote-server/remote_server.pdb -Destination $remoteDest
            "#})
            .if_condition(Expression::new("matrix.platform == 'windows'"))
            .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
            .add_env(("TARGET", "${{ matrix.target }}"))
            .add_env(("REMOTE_TARGET", "${{ matrix.remote_target }}"))
            .add_env(("APPLICATION_FEATURES", "${{ matrix.application_features }}"))
            .add_env(("REMOTE_FEATURES", "${{ matrix.remote_features }}"))
            .add_env(("COMMIT_SHA", prepared.commit_sha.to_string()))
        )
        .add_step(
            cache_macos_cargo_bundle()
                .if_condition(Expression::new("matrix.platform == 'macos'")),
        )
        .add_step(
            named::bash("uname -m && rustc -vV && df -h . && xcode-select -p")
                .if_condition(Expression::new("matrix.platform == 'macos'")),
        )
        .add_step(
            steps::enable_windows_long_paths()
                .if_condition(Expression::new("matrix.platform == 'windows'")),
        )
        .add_step(
            named::bash("./script/linux && ./script/download-wasi-sdk")
                .if_condition(Expression::new("matrix.platform == 'linux'")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                cargo xtask bundle --product rust --platform linux --target "$TARGET" --phase package --signing required
            "#})
                .if_condition(Expression::new("matrix.platform == 'linux'"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("CC", "clang"))
                .add_env(("CXX", "clang++"))
                .add_env(("LINUX_SIGNING_KEY", "${{ secrets.LINUX_SIGNING_KEY }}"))
                .add_env(("LINUX_SIGNING_KEY_PASSWORD", "${{ secrets.LINUX_SIGNING_KEY_PASSWORD }}"))
                .add_env(("LINUX_SIGNING_KEY_FINGERPRINT", "${{ secrets.LINUX_SIGNING_KEY_FINGERPRINT }}")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                cargo xtask bundle --product rust --platform macos --target "$TARGET" --phase package --signing required
            "#})
                .if_condition(Expression::new("matrix.platform == 'macos'"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("MACOS_SIGNING_IDENTITY", "${{ secrets.MACOS_SIGNING_IDENTITY }}"))
                .add_env(("MACOS_CERTIFICATE", "${{ secrets.MACOS_CERTIFICATE }}"))
                .add_env(("MACOS_CERTIFICATE_PASSWORD", "${{ secrets.MACOS_CERTIFICATE_PASSWORD }}"))
                .add_env(("APPLE_NOTARIZATION_KEY", "${{ secrets.APPLE_NOTARIZATION_KEY }}"))
                .add_env(("APPLE_NOTARIZATION_KEY_ID", "${{ secrets.APPLE_NOTARIZATION_KEY_ID }}"))
                .add_env(("APPLE_NOTARIZATION_ISSUER_ID", "${{ secrets.APPLE_NOTARIZATION_ISSUER_ID }}")),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                cargo xtask bundle --product rust --platform windows --target $env:TARGET --phase package --signing required
                if ($LASTEXITCODE -ne 0) { throw "Product-aware Windows bundling failed" }
            "#})
                .if_condition(Expression::new("matrix.platform == 'windows'"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("AZURE_TENANT_ID", "${{ secrets.AZURE_TENANT_ID }}"))
                .add_env(("AZURE_CLIENT_ID", "${{ secrets.AZURE_CLIENT_ID }}"))
                .add_env(("AZURE_CLIENT_SECRET", "${{ secrets.AZURE_CLIENT_SECRET }}"))
                .add_env(("ACCOUNT_NAME", "${{ secrets.ACCOUNT_NAME }}"))
                .add_env(("CERT_PROFILE_NAME", "${{ secrets.CERT_PROFILE_NAME }}"))
                .add_env(("ENDPOINT", "${{ secrets.ENDPOINT }}"))
                .add_env(("WINDOWS_SIGNING_PUBLISHER", "${{ secrets.WINDOWS_SIGNING_PUBLISHER }}"))
                .add_env(("FILE_DIGEST", "${{ secrets.FILE_DIGEST }}"))
                .add_env(("TIMESTAMP_DIGEST", "${{ secrets.TIMESTAMP_DIGEST }}"))
                .add_env(("TIMESTAMP_SERVER", "${{ secrets.TIMESTAMP_SERVER }}")),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                export ZED_PRODUCT_ARTIFACT_NAME="${PRODUCT_ID}-${RELEASE_VERSION}-${ARTIFACT_SUFFIX}"
                python3 script/smoke-product-bundle \
                    --product "$PRODUCT_ID" \
                    --platform "$PLATFORM" \
                    --target "$TARGET" \
                    --plan "release-bundle-plans/$PLAN_FILE" \
                    --commit-sha "$COMMIT_SHA"
            "#})
                .if_condition(Expression::new("matrix.platform != 'windows'"))
                .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
                .add_env(("PLATFORM", "${{ matrix.platform }}"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("PLAN_FILE", "${{ matrix.plan_file }}"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("MACOS_SIGNING_IDENTITY", "${{ secrets.MACOS_SIGNING_IDENTITY }}"))
                .add_env(("LINUX_SIGNING_KEY_FINGERPRINT", "${{ secrets.LINUX_SIGNING_KEY_FINGERPRINT }}"))
                .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                $env:ZED_PRODUCT_ARTIFACT_NAME = "$env:PRODUCT_ID-$env:RELEASE_VERSION-$env:ARTIFACT_SUFFIX"
                python script/smoke-product-bundle `
                    --product $env:PRODUCT_ID `
                    --platform windows `
                    --target $env:TARGET `
                    --plan "release-bundle-plans/$env:PLAN_FILE" `
                    --commit-sha $env:COMMIT_SHA
            "#})
                .if_condition(Expression::new("matrix.platform == 'windows'"))
                .add_env(("PRODUCT_ID", "${{ matrix.product }}"))
                .add_env(("TARGET", "${{ matrix.target }}"))
                .add_env(("PLAN_FILE", "${{ matrix.plan_file }}"))
                .add_env(("ARTIFACT_SUFFIX", "${{ matrix.artifact_suffix }}"))
                .add_env(("WINDOWS_SIGNING_PUBLISHER", "${{ secrets.WINDOWS_SIGNING_PUBLISHER }}"))
                .add_env(("COMMIT_SHA", prepared.commit_sha.to_string())),
        )
        .add_step(
            steps::upload_artifact("${{ matrix.artifact }}-signing-evidence", "target/products/${{ matrix.product }}/signing-evidence")
                .if_condition(Expression::new("always() && matrix.platform == 'macos'")),
        )
        .add_step(
            steps::upload_artifact("${{ matrix.artifact }}", "${{ matrix.artifact_path }}")
                .if_no_files_found(IfNoFilesFound::Error),
        )
        .add_step(
            named::bash(indoc::indoc! {r#"
                printf '### sccache: package-only job\n\nOnly the xtask orchestrator is compiled here; this job consumed commit-bound application, remote-server, license, and bundle-plan artifacts. Cache requests: 0.\n' >> "$GITHUB_STEP_SUMMARY"
            "#})
                .if_condition(Expression::new("always() && matrix.platform != 'windows'")),
        )
        .add_step(
            named::pwsh(indoc::indoc! {r#"
                Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value '### sccache: package-only job'
                Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value ''
                Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value 'Only the xtask orchestrator is compiled here; this job consumed commit-bound application, remote-server, license, and bundle-plan artifacts. Cache requests: 0.'
            "#})
                .if_condition(Expression::new("always() && matrix.platform == 'windows'")),
        )
        .add_step(
            named::bash("df -h . && du -sh target")
                .if_condition(Expression::new("always() && matrix.platform == 'macos'")),
        );

    named::job(release_bundle_environment(job))
}

fn publish_product(prepare: &NamedJob, prepared: &PreparedRelease, build: &NamedJob) -> NamedJob {
    let manifest = ProductManifest::load().expect("product catalog must be valid");
    let product = manifest.product("rust").expect("Rust product must exist");
    let artifact_patterns = product
        .targets
        .iter()
        .map(|target| match target.as_str() {
            "linux-x86_64" => "'rust-*-linux-x86_64.tar.gz'\n    'rust-*-linux-x86_64.tar.gz.asc'\n    'rust-*-linux-x86_64.tar.gz.pub'",
            "macos-aarch64" => "'rust-*-macos-aarch64.dmg'",
            "windows-x86_64" => "'rust-*-windows-x86_64.exe'",
            unsupported => panic!("validated catalog contains unsupported target {unsupported}"),
        })
        .collect::<Vec<_>>()
        .join("\n    ");
    let expected_count = artifact_patterns.lines().count();
    let publish = named::bash(
        formatdoc! {r#"
        git fetch origin +refs/heads/release:refs/remotes/origin/release --tags

        ARGS=(release-version --commit "$COMMIT_SHA" --bootstrap-commit "$BOOTSTRAP_SHA" --bump "$BUMP")
        if [[ -n "$EXPLICIT_VERSION" ]]; then
            ARGS+=(--explicit-version "$EXPLICIT_VERSION")
        fi
        DECISION="$(cargo xtask "${{ARGS[@]}}")"
        ACTUAL_COMMIT="$(jq -r '.commit_sha' <<<"$DECISION")"
        ACTUAL_VERSION="$(jq -r '.version' <<<"$DECISION")"
        ACTUAL_TAG="$(jq -r '.tag' <<<"$DECISION")"
        if [[ "$ACTUAL_COMMIT" != "$COMMIT_SHA" || "$ACTUAL_VERSION" != "$VERSION" || "$ACTUAL_TAG" != "$TAG" ]]; then
            echo "Release state changed while artifacts were building; refusing stale publication" >&2
            exit 1
        fi

        mapfile -t FILES < <(find release-artifacts -type f -print | sort)
        if [[ ${{#FILES[@]}} -ne {expected_count} ]]; then
            echo "Expected exactly {expected_count} Copper artifacts, found ${{#FILES[@]}}" >&2
            exit 1
        fi
        PATTERNS=(
            {artifact_patterns}
        )
        for PATTERN in "${{PATTERNS[@]}}"; do
            mapfile -t MATCHES < <(find release-artifacts -type f -name "$PATTERN")
            if [[ ${{#MATCHES[@]}} -ne 1 ]]; then
                echo "Expected exactly one artifact matching $PATTERN, found ${{#MATCHES[@]}}" >&2
                exit 1
            fi
        done
        LINUX_ARCHIVE="$(find release-artifacts -type f -name 'rust-*-linux-x86_64.tar.gz')"
        python3 script/sign-linux-archive --verify-only "$LINUX_ARCHIVE"
        if git show-ref --verify --quiet "refs/tags/$TAG"; then
            TAG_COMMIT="$(git rev-parse "$TAG^{{commit}}")"
            if [[ "$TAG_COMMIT" != "$COMMIT_SHA" ]]; then
                echo "Existing tag $TAG points to $TAG_COMMIT, not $COMMIT_SHA" >&2
                exit 1
            fi
        else
            git config user.name "github-actions[bot]"
            git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
            git tag -a "$TAG" "$COMMIT_SHA" -m "{display_name} $VERSION"
            git push origin "refs/tags/$TAG"
        fi

        TITLE="{display_name} $VERSION"
        if ! gh release view "$TAG" >/dev/null 2>&1; then
            gh release create "$TAG" --draft --title "$TITLE" --generate-notes --verify-tag
        fi

        mapfile -t REMOTE_NAMES < <(gh release view "$TAG" --json assets --jq '.assets[].name' | sort)
        for FILE in "${{FILES[@]}}"; do
            NAME="$(basename "$FILE")"
            if ! printf '%s\n' "${{REMOTE_NAMES[@]}}" | grep -Fxq "$NAME"; then
                gh release upload "$TAG" "$FILE"
            fi
        done

        mapfile -t EXPECTED_NAMES < <(printf '%s\n' "${{FILES[@]}}" | xargs -n1 basename | sort)
        mapfile -t REMOTE_NAMES < <(gh release view "$TAG" --json assets --jq '.assets[].name' | sort)
        if ! diff -u <(printf '%s\n' "${{EXPECTED_NAMES[@]}}") <(printf '%s\n' "${{REMOTE_NAMES[@]}}") ; then
            echo "GitHub Release assets do not exactly match the validated Copper artifacts" >&2
            exit 1
        fi
        gh release edit "$TAG" --title "$TITLE" --draft=false
    "#,
            display_name = product.display_name,
        },
    )
    .add_env(("COMMIT_SHA", prepared.commit_sha.to_string()))
    .add_env(("VERSION", prepared.version.to_string()))
    .add_env(("TAG", prepared.tag.to_string()))
    .add_env(("BUMP", RELEASE_BUMP))
    .add_env(("EXPLICIT_VERSION", RELEASE_EXPLICIT_VERSION))
    .add_env(("BOOTSTRAP_SHA", "${{ vars.COPPER_RELEASE_BOOTSTRAP_SHA }}"))
    .add_env(("GH_TOKEN", vars::GITHUB_TOKEN))
    .add_env(("LINUX_SIGNING_KEY_FINGERPRINT", "${{ secrets.LINUX_SIGNING_KEY_FINGERPRINT }}"));

    let mut job = Job::default()
        .needs([prepare.name.clone(), build.name.clone()])
        .runs_on("ubuntu-22.04")
        .timeout_minutes(15u32)
        .permissions(
            Permissions::default()
                .contents(Level::Write)
                .actions(Level::Read),
        )
        .add_step(
            steps::checkout_repo()
                .with_ref(prepared.commit_sha.to_string())
                .with_full_history()
                .with_fetch_tags(),
        );
    for target in &product.targets {
        let artifact = match target.as_str() {
            "linux-x86_64" => "rust-linux-x86_64",
            "macos-aarch64" => "rust-macos-aarch64",
            "windows-x86_64" => "rust-windows-x86_64",
            unsupported => panic!("validated catalog contains unsupported target {unsupported}"),
        };
        job = job.add_step(
            steps::download_artifact()
                .artifact_name(artifact)
                .path("release-artifacts"),
        );
    }
    named::job(
        job.add_step(promotion_check().add_env(("COMMIT_SHA", prepared.commit_sha.to_string())))
            .add_step(publish),
    )
}

fn refresh_website(publish: &NamedJob, prepare: &NamedJob, prepared: &PreparedRelease) -> NamedJob {
    // GITHUB_TOKEN publication events cannot start a follow-up workflow.
    named::job(
        super::website_refresh::refresh_job(prepared.tag.to_string())
            .needs([publish.name.clone(), prepare.name.clone()]),
    )
}

#[cfg(test)]
mod release_workflow_tests {
    use super::*;

    #[test]
    fn website_refresh_follows_publication_without_checkout_or_token_events() -> anyhow::Result<()>
    {
        let workflow = serde_json::to_value(release_promotion())?;
        let refresh = &workflow["jobs"]["refresh_website"];
        assert_eq!(
            refresh["needs"],
            json!(["publish_product", "prepare_release"])
        );
        assert_eq!(refresh["permissions"]["contents"], "read");
        let step = &refresh["steps"][0];
        assert_eq!(
            step["env"]["AMPLIFY_REBUILD_WEBHOOK"],
            "${{ secrets.AMPLIFY_REBUILD_WEBHOOK }}"
        );
        assert_eq!(step["shell"], "bash -euo pipefail {0}");
        assert!(step.get("uses").is_none());
        let script = step["run"].as_str().unwrap_or_default();
        assert!(script.contains("method=\"POST\""));
        assert!(script.contains("NoRedirects"));
        assert!(!script.contains("print(url)"));
        assert!(script.contains("wait_for_deployment"));
        assert_eq!(
            step["env"]["COPPER_RELEASE_TAG"],
            "${{ needs.prepare_release.outputs.tag }}"
        );
        Ok(())
    }

    fn yaml() -> String {
        release_promotion()
            .to_string()
            .expect("release workflow must serialize")
    }

    #[test]
    fn signing_check_reuses_production_packages_without_a_publication_path() -> anyhow::Result<()> {
        let check: serde_yaml::Value = serde_yaml::from_str(
            &release_signing_check()
                .to_string()
                .map_err(|error| anyhow::anyhow!("{error:?}"))?,
        )?;
        let production: serde_yaml::Value = serde_yaml::from_str(&yaml())?;
        let check_jobs = check["jobs"].as_mapping().expect("validation jobs");
        assert_eq!(check_jobs.len(), 6);
        assert!(check["on"]["workflow_dispatch"].is_mapping());
        assert_eq!(
            check["on"]["push"]["branches"],
            serde_yaml::to_value(["codex/**", "main"])?
        );
        assert!(
            check["on"]["push"]["paths"]
                .as_sequence()
                .expect("contract paths")
                .contains(&serde_yaml::Value::String(
                    ".github/workflows/release_signing_check.yml".to_owned()
                ))
        );
        assert!(check["on"]["workflow_run"].is_null());
        let mut preflight = check["jobs"]["signing_preflight"].clone();
        let condition = preflight
            .as_mapping_mut()
            .expect("preflight job")
            .remove(serde_yaml::Value::String("if".to_owned()));
        assert_eq!(
            condition,
            Some(serde_yaml::Value::String(
                "github.event_name == 'workflow_dispatch' && inputs.sign_artifacts == true"
                    .to_owned()
            ))
        );
        assert_eq!(preflight, production["jobs"]["signing_preflight"]);
        assert_eq!(
            check["on"]["workflow_dispatch"]["inputs"]["sign_artifacts"]["default"],
            "false"
        );
        assert_eq!(
            check["on"]["workflow_dispatch"]["inputs"]["sign_artifacts"]["type"],
            "boolean"
        );
        for job in [
            "release_licenses",
            "application_builds",
            "remote_server_builds",
            "product_builds",
        ] {
            assert_eq!(check["jobs"][job], production["jobs"][job]);
        }
        for job in ["publish_product", "refresh_website"] {
            assert!(check["jobs"][job].is_null());
        }
        for job in check_jobs.values() {
            assert_eq!(job["permissions"]["contents"], "read");
        }
        assert_eq!(
            check["on"]["workflow_dispatch"]["inputs"]["version"]["default"],
            "0.1.0"
        );
        assert_eq!(
            production["jobs"]["product_builds"]["env"]["ZED_PRODUCT_SIGNING"],
            "required"
        );
        assert_eq!(
            production["jobs"]["release_licenses"]["needs"],
            serde_yaml::to_value(["prepare_release", "signing_preflight"])?
        );
        Ok(())
    }

    #[test]
    fn successful_release_ci_is_the_only_automatic_release_path() {
        let yaml = yaml();
        assert!(yaml.contains("workflow_run:"));
        assert!(yaml.contains("- run_tests"));
        assert!(yaml.contains("- completed"));
        assert!(yaml.contains("- release"));
        assert!(!yaml.contains("- main"));
        assert!(yaml.contains("github.event.workflow_run.conclusion == 'success'"));
        assert!(yaml.contains("github.event.workflow_run.event == 'push'"));
        assert!(yaml.contains("github.event.workflow_run.head_branch == 'release'"));
        assert!(
            yaml.contains(
                "github.event.workflow_run.head_repository.full_name == github.repository"
            )
        );
        assert!(yaml.contains("github.event.workflow_run.head_sha"));
    }

    #[test]
    fn manual_release_inputs_require_release_branch_without_tag_trigger() {
        let yaml = yaml();
        assert!(yaml.contains("bump:"));
        assert!(yaml.contains("default: patch"));
        assert!(yaml.contains("version:"));
        assert!(yaml.contains("commit_sha:"));
        assert!(yaml.contains("--explicit-version"));
        let workflow =
            serde_json::to_value(release_promotion()).expect("serialize release workflow");
        assert!(workflow["on"]["push"].is_null());
        assert!(workflow["on"]["pull_request"].is_null());
        assert!(!yaml.contains("startsWith(github.ref, 'refs/tags/"));
        assert!(yaml.contains("github.ref == 'refs/heads/release'"));
    }

    #[test]
    fn publication_is_serial_and_depends_on_every_build() {
        let yaml = yaml();
        assert!(yaml.contains("group: copper-release"));
        assert!(yaml.contains("cancel-in-progress: false"));
        assert!(yaml.contains("needs:\n    - prepare_release\n    - product_builds"));
        assert!(yaml.contains("if-no-files-found: error"));
        let smoke = yaml
            .find("script/smoke-product-bundle")
            .expect("native bundle smoke check");
        let upload = yaml
            .rfind("uses: actions/upload-artifact")
            .expect("bundle upload");
        assert!(smoke < upload);
        assert!(yaml.contains("Expected exactly 5 Copper artifacts"));
        assert!(yaml.contains("git config --global core.longpaths true"));
        assert!(yaml.contains("CC: clang\n"));
        assert!(yaml.contains("CXX: clang++\n"));
        assert!(yaml.contains(
            "uses: mozilla-actions/sccache-action@fc920bf0ec8de6ee65d409111f7ec508035751ba"
        ));
        assert!(yaml.contains("SCCACHE_GHA_ENABLED: on"));
        assert!(yaml.contains("SCCACHE_GHA_VERSION=release-v2-"));
        assert!(yaml.contains("CACHE_TARGET: ${{ matrix.remote_target }}"));
        assert!(yaml.contains("SCCACHE_IDLE_TIMEOUT: '0'"));
        assert!(!yaml.contains("SCCACHE_LOCAL_CACHE_DIR"));
        assert_eq!(yaml.matches("uses: actions/cache@").count(), 1);
        assert!(yaml.contains("path: ~/.cargo/bin/cargo-bundle"));
        assert!(yaml.contains(
            "key: release-tool-v1-cargo-bundle-0.11.0-${{ runner.os }}-${{ runner.arch }}"
        ));
        assert!(yaml.contains("sccache --show-stats"));
        assert!(yaml.contains("sccache --stop-server"));
        assert!(yaml.contains("name: release-bundle-plans"));
        assert!(yaml.contains("target/debug/xtask bundle --product rust --platform linux"));
        assert!(yaml.contains("target/debug/xtask bundle --product rust --platform macos"));
        assert!(yaml.contains("target/debug/xtask bundle --product rust --platform windows"));
        assert!(yaml.contains("'. + {commit_sha: $commit_sha}'"));
        assert!(yaml.contains("--plan \"release-bundle-plans/$PLAN_FILE\""));
        assert!(yaml.contains("--commit-sha \"$COMMIT_SHA\""));
        assert!(!yaml.contains("cargo xtask bundle --product $env:PRODUCT_ID"));
        assert!(yaml.contains("cargo-about@0.8.2"));
        assert!(yaml.contains("fallback: none"));
        assert!(yaml.contains("sha256sum \"$1\""));
        assert!(yaml.contains("shasum -a 256 \"$1\""));
        assert!(yaml.contains("git diff --quiet \"$COMMIT_SHA\" -- Cargo.lock"));
        assert!(yaml.contains("git rev-parse \"${COMMIT_SHA}:Cargo.lock\""));
        assert!(yaml.contains("cargo_lock_blob"));
        assert!(
            yaml.contains("ZED_PRODUCT_ICON_SET: ${{ github.workspace }}/${{ matrix.icon_set }}")
        );
        assert!(!yaml.contains("CARGO_LOCK_HASH=\"$(sha256_file Cargo.lock)\""));
        assert!(!yaml.contains("LICENSES_HASH=\"$(git hash-object"));
        assert!(yaml.contains("export ZED_BUNDLE_PHASE=application"));
        assert!(yaml.contains("export ZED_BUNDLE_PHASE=remote-server"));
        for platform in ["linux", "macos", "windows"] {
            assert!(yaml.contains(&format!(
                "cargo xtask bundle --product rust --platform {platform}"
            )));
        }
        assert!(yaml.contains("--phase package --signing required"));
        assert!(yaml.contains("script/bundle-windows.ps1 -Architecture x86_64"));
        assert!(!yaml.contains("cargo xtask bundle --phase"));
        assert!(yaml.contains(
            "needs:\n    - prepare_release\n    - release_licenses\n    - application_builds\n    - remote_server_builds"
        ));
        assert!(yaml.contains("git tag -a"));
        assert!(yaml.contains("--draft"));
        assert!(yaml.contains("contents: write"));

        let package_job = yaml
            .split_once("  product_builds:\n")
            .expect("package job")
            .1
            .split_once("  publish_product:\n")
            .expect("publisher after package job")
            .0;
        assert!(!package_job.contains("mozilla-actions/sccache-action"));
        assert!(!package_job.contains("RUSTC_WRAPPER"));
        assert!(!package_job.contains("sccache --show-stats"));
        assert!(package_job.contains("Cache requests: 0."));
    }
}

pub(crate) struct ReleaseBundleJobs {
    pub linux_aarch64: NamedJob,
    pub linux_x86_64: NamedJob,
    pub bwrap_linux_aarch64: NamedJob,
    pub bwrap_linux_x86_64: NamedJob,
    pub mac_aarch64: NamedJob,
    pub mac_x86_64: NamedJob,
    pub windows_aarch64: NamedJob,
    pub windows_x86_64: NamedJob,
}

impl ReleaseBundleJobs {
    pub fn jobs(&self) -> Vec<&NamedJob> {
        vec![
            &self.linux_aarch64,
            &self.linux_x86_64,
            &self.bwrap_linux_aarch64,
            &self.bwrap_linux_x86_64,
            &self.mac_aarch64,
            &self.mac_x86_64,
            &self.windows_aarch64,
            &self.windows_x86_64,
        ]
    }

    pub fn into_jobs(self) -> Vec<NamedJob> {
        vec![
            self.linux_aarch64,
            self.linux_x86_64,
            self.bwrap_linux_aarch64,
            self.bwrap_linux_x86_64,
            self.mac_aarch64,
            self.mac_x86_64,
            self.windows_aarch64,
            self.windows_x86_64,
        ]
    }
}

pub(crate) fn create_sentry_release() -> Step<Use> {
    named::uses(
        "getsentry",
        "action-release",
        "526942b68292201ac6bbb99b9a0747d4abee354c", // v3
    )
    .add_env(("SENTRY_ORG", "zed-dev"))
    .add_env(("SENTRY_PROJECT", "zed"))
    .add_env(("SENTRY_AUTH_TOKEN", vars::SENTRY_AUTH_TOKEN))
    .add_with(("environment", "production"))
}

pub(crate) const COMPLIANCE_REPORT_PATH: &str = "compliance-report-${GITHUB_REF_NAME}.md";
pub(crate) const COMPLIANCE_REPORT_ARTIFACT_PATH: &str =
    "compliance-report-${{ github.ref_name }}.md";
pub(crate) const COMPLIANCE_STEP_ID: &str = "run-compliance-check";
const NEEDS_REVIEW_PULLS_URL: &str = "https://github.com/imagine-engineering/made/pulls?q=is%3Apr+is%3Aclosed+label%3A%22PR+state%3Aneeds+review%22";

pub(crate) enum ComplianceContext {
    Scheduled { tag_source: StepOutput },
}

pub(crate) fn add_compliance_steps(
    job: gh_workflow::Job,
    context: ComplianceContext,
) -> (gh_workflow::Job, StepOutput) {
    let ComplianceContext::Scheduled { tag_source } = context;
    let tag_source = tag_source.to_string();
    let compliance_step = named::bash(formatdoc! {r#"
        cargo xtask compliance version "$LATEST_TAG" --branch main --report-path "{COMPLIANCE_REPORT_PATH}"
    "#})
    .id(COMPLIANCE_STEP_ID)
    .add_env(("GITHUB_APP_ID", vars::ZED_ZIPPY_APP_ID))
    .add_env(("GITHUB_APP_KEY", vars::ZED_ZIPPY_APP_PRIVATE_KEY))
    .add_env(("LATEST_TAG", tag_source.clone()))
    .continue_on_error(true);
    let check_result = StepOutput::new_unchecked(&compliance_step, "outcome");

    let notification_script = formatdoc! {r#"
        if [ "$COMPLIANCE_OUTCOME" == "success" ]; then
            STATUS="✅ Scheduled compliance check passed for $COMPLIANCE_TAG"
            MESSAGE=$(printf "%s\n\nReport: %s" "$STATUS" "$ARTIFACT_URL")
        else
            STATUS="⚠️ Scheduled compliance check failed for $COMPLIANCE_TAG"
            MESSAGE=$(printf "%s\n\nReport: %s\nPRs needing review: %s" "$STATUS" "$ARTIFACT_URL" "{NEEDS_REVIEW_PULLS_URL}")
        fi

        curl -X POST -H 'Content-type: application/json' \
            --data "$(jq -n --arg text "$MESSAGE" '{{"text": $text}}')" \
            "$SLACK_WEBHOOK"
    "#};

    let notification_step = Step::new("send_compliance_slack_notification")
        .run(notification_script)
        .if_condition(Expression::new("${{ always() }}"))
        .add_env(("SLACK_WEBHOOK", vars::SLACK_WEBHOOK_WORKFLOW_FAILURES))
        .add_env((
            "COMPLIANCE_OUTCOME",
            format!("${{{{ steps.{COMPLIANCE_STEP_ID}.outcome }}}}"),
        ))
        .add_env(("COMPLIANCE_TAG", tag_source))
        .add_env((
            "ARTIFACT_URL",
            format!("{CURRENT_ACTION_RUN_URL}#artifacts"),
        ));

    (
        job.add_step(compliance_step)
            .add_step(
                upload_artifact(COMPLIANCE_REPORT_ARTIFACT_PATH)
                    .if_condition(Expression::new("always()")),
            )
            .add_step(notification_step),
        check_result,
    )
}

pub(crate) fn download_workflow_artifacts() -> DownloadArtifactStep {
    steps::download_artifact().path("./artifacts/")
}

pub(crate) fn prep_release_artifacts() -> Step<Run> {
    let mut script_lines = vec!["mkdir -p release-artifacts/\n".to_string()];
    for asset in assets::all() {
        let mv_command = format!("mv ./artifacts/{asset}/{asset} release-artifacts/{asset}");
        script_lines.push(mv_command)
    }

    named::bash(&script_lines.join("\n"))
}

pub(crate) fn notify_on_failure(deps: &[&NamedJob]) -> NamedJob {
    let failure_message = format!("❌ ${{{{ github.workflow }}}} failed: {CURRENT_ACTION_RUN_URL}");
    let notification = named::bash(
        r#"curl -X POST -H 'Content-type: application/json' --data "$(jq -n --arg text "$SLACK_MESSAGE" '{"text": $text}')" "$SLACK_WEBHOOK""#,
    )
    .add_env(("SLACK_WEBHOOK", vars::SLACK_WEBHOOK_WORKFLOW_FAILURES))
    .add_env(("SLACK_MESSAGE", failure_message));

    named::job(
        dependant_job(deps)
            .runs_on(runners::LINUX_SMALL)
            .cond(Expression::new("failure()"))
            .add_step(notification),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_product_release_builds_then_publishes() -> anyhow::Result<()> {
        let workflow = release_promotion()
            .to_string()
            .map_err(|error| anyhow::anyhow!("failed to serialize release workflow: {error:?}"))?;

        assert_eq!(workflow.matches("runs-on:").count(), 8);
        assert!(workflow.contains("runner: ubuntu-22.04"));
        assert!(workflow.contains("runner: macos-15"));
        assert!(workflow.contains("runner: windows-2022"));
        assert!(workflow.contains("cargo xtask bundle --product rust --platform"));
        assert!(workflow.contains("--phase package --signing required"));
        assert!(!workflow.contains("cargo xtask bundle --phase"));
        assert!(workflow.contains("product: rust"));
        assert!(workflow.contains("target: x86_64-unknown-linux-gnu"));
        assert!(workflow.contains("target: aarch64-apple-darwin"));
        assert!(workflow.contains("target: x86_64-pc-windows-msvc"));
        assert!(workflow.contains("git config --global core.longpaths true"));
        assert!(workflow.contains("CC: clang"));
        assert!(workflow.contains("CXX: clang++"));
        assert!(workflow.contains("needs:\n    - prepare_release\n    - product_builds"));
        assert_eq!(workflow.matches("contents: write").count(), 1);
        let rust = ProductManifest::load()?
            .product("rust")?
            .display_name
            .clone();
        assert!(workflow.contains(&format!("TITLE=\"{rust} $VERSION\"")));

        for artifact in [
            "rust-*-linux-x86_64.tar.gz",
            "rust-*-macos-aarch64.dmg",
            "rust-*-windows-x86_64.exe",
        ] {
            assert!(workflow.contains(artifact), "missing {artifact}");
        }

        for forbidden in [
            "simtropolis/zed",
            "repository_owner",
            "namespace-profile",
            "self-32vcpu",
            "R2_ACCOUNT_ID",
            "SENTRY_AUTH_TOKEN",
            "SLACK_WEBHOOK",
            "compliance",
            "Rustlings",
        ] {
            assert!(
                !workflow.contains(forbidden),
                "found {forbidden} in release"
            );
        }

        Ok(())
    }
}
