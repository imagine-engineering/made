use std::time::Duration;

use anyhow::{Context as _, Result, ensure};
use feature_flags::{
    FeatureFlagAppExt as _, RustNamedCargoPresetsExperimentalFeatureFlag,
    RustNativeProfilingExperimentalFeatureFlag,
};
use gpui::{App, BackgroundExecutor, FutureExt as _, Task};
use rpc::{AnyProtoClient, proto};
use uuid::Uuid;

use crate::{
    Project, ProjectPath, cargo_workspace::CargoTargetKind,
    cargo_workspace_store::HostCargoProfileSelection, profile_results_store::RemoteProfileError,
    trusted_worktrees::TrustedWorktrees,
};
use proto::ProfileResultStatus as Status;

pub struct RemoteLinuxCargoProfileReview {
    operation: RemoteProfileOperation,
    token: Vec<u8>,
    paths: [ProjectPath; 2],
}

pub struct RemoteLinuxCargoProfileResult {
    pub run_id: Uuid,
    pub paths: [ProjectPath; 2],
}

struct RemoteProfileOperation {
    client: AnyProtoClient,
    executor: BackgroundExecutor,
    id: Uuid,
    active: bool,
}

impl Drop for RemoteProfileOperation {
    fn drop(&mut self) {
        if self.active {
            let request = proto::CancelLinuxCargoProfile {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                operation_id: self.id.as_bytes().to_vec(),
            };
            let client = self.client.clone();
            let executor = self.executor.clone();
            self.executor
                .spawn(async move {
                    match client
                        .request(request)
                        .with_timeout(Duration::from_secs(3), &executor)
                        .await
                    {
                        Ok(Ok(_)) => {}
                        Ok(Err(error)) => log::debug!(
                            "Remote Cargo profile cancellation could not be sent: {error:#}"
                        ),
                        Err(error) => {
                            log::debug!("Waiting for remote Cargo profile cancellation: {error:#}")
                        }
                    }
                })
                .detach();
        }
    }
}

impl Project {
    pub fn is_direct_linux_profile_host(&self, cx: &App) -> bool {
        cfg!(any(target_os = "macos", target_os = "linux"))
            && !self.is_shared()
            && self.remote_client().is_some_and(|client| {
                client.read(cx).remote_platform().os == remote::RemoteOs::Linux
            })
    }

    pub fn prepare_remote_linux_cargo_profile(
        &self,
        selection: HostCargoProfileSelection,
        operation_id: Uuid,
        cx: &App,
    ) -> Result<Task<Result<RemoteLinuxCargoProfileReview>>> {
        ensure!(
            self.is_direct_linux_profile_host(cx),
            RemoteProfileError::Unsupported
        );
        ensure!(!self.is_disconnected(cx), RemoteProfileError::Disconnected);
        ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
                && cx.has_flag::<RustNamedCargoPresetsExperimentalFeatureFlag>()
                && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store(), cx),
            "Enable native profiling and named Cargo presets and trust the project before capture"
        );
        let client = self
            .remote_client()
            .context("The Linux project host disconnected")?
            .read(cx)
            .proto_client();
        let target_kind = match selection.target_kind {
            CargoTargetKind::Binary => proto::LinuxCargoProfileTargetKind::Binary,
            CargoTargetKind::Example => proto::LinuxCargoProfileTargetKind::Example,
            _ => anyhow::bail!("Native profiling requires a binary or example target"),
        };
        let executor = cx.background_executor().clone();
        Ok(cx.background_executor().spawn(async move {
            let capability = client
                .request(proto::GetLinuxCargoProfileSupport {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                })
                .with_timeout(Duration::from_secs(3), &executor)
                .await
                .map_err(|_| RemoteProfileError::Unsupported)??;
            check_profile_response(
                &capability,
                &[],
                "The Linux host could not prepare native profiling",
            )?;
            let operation = RemoteProfileOperation {
                client,
                executor,
                id: operation_id,
                active: true,
            };
            let selection_root = selection.workspace.root;
            let reply = operation
                .client
                .request(proto::PrepareLinuxCargoProfile {
                    project_id: proto::REMOTE_SERVER_PROJECT_ID,
                    operation_id: operation_id.as_bytes().to_vec(),
                    workspace_root: Some(selection_root.to_proto()),
                    package_manifest: Some(selection.package_manifest.to_proto()),
                    target_kind: target_kind as i32,
                    target_name: selection.target_name,
                    preset_id: selection.preset_id,
                })
                .with_timeout(Duration::from_secs(30 * 60), &operation.executor)
                .await
                .context("The Linux Cargo profile preparation timed out")??;
            check_profile_response(
                &reply,
                operation_id.as_bytes(),
                "The host could not prepare the selected Cargo preset",
            )?;
            ensure!(
                Uuid::from_slice(&reply.review_token).is_ok_and(|token| !token.is_nil())
                    && reply.task_run_id.is_empty(),
                "The host returned an invalid Cargo profile review"
            );
            let paths = profile_response_paths(reply.profile_path, reply.symbols_path)?;
            ensure!(
                paths.iter().all(|path| path.starts_with(&selection_root)),
                "The host profile outputs are outside the selected Cargo workspace"
            );
            Ok(RemoteLinuxCargoProfileReview {
                operation,
                token: reply.review_token,
                paths,
            })
        }))
    }
}

impl RemoteLinuxCargoProfileReview {
    pub fn paths(&self) -> &[ProjectPath; 2] {
        &self.paths
    }

    pub async fn capture(mut self) -> Result<RemoteLinuxCargoProfileResult> {
        let reply = self
            .operation
            .client
            .request(proto::FinishLinuxCargoProfileReview {
                project_id: proto::REMOTE_SERVER_PROJECT_ID,
                operation_id: self.operation.id.as_bytes().to_vec(),
                review_token: self.token.clone(),
                approved: true,
            })
            .with_timeout(Duration::from_secs(120), &self.operation.executor)
            .await
            .context("The Linux Cargo profile capture timed out")??;
        check_profile_response(
            &reply,
            self.operation.id.as_bytes(),
            "The host could not capture the selected Cargo profile",
        )?;
        let run_id = Uuid::parse_str(&reply.task_run_id)
            .context("The host returned an invalid Cargo profile run")?;
        let paths = profile_response_paths(reply.profile_path, reply.symbols_path)?;
        ensure!(
            reply.review_token.is_empty() && !run_id.is_nil() && paths == self.paths,
            "The host returned outputs that differ from the reviewed Cargo profile"
        );
        self.operation.active = false;
        Ok(RemoteLinuxCargoProfileResult { run_id, paths })
    }
}

fn check_profile_response(
    reply: &proto::LinuxCargoProfileResponse,
    operation: &[u8],
    failure: &'static str,
) -> Result<()> {
    ensure!(
        reply.native_receiver == 0 && reply.operation_id == operation,
        "The remote Cargo profile response does not belong to this request"
    );
    match reply.status() {
        Status::Ready => Ok(()),
        Status::Cancelled => Err(RemoteProfileError::Cancelled.into()),
        Status::Stale => Err(RemoteProfileError::Stale.into()),
        Status::Restricted => Err(RemoteProfileError::Restricted.into()),
        Status::Unsupported => Err(RemoteProfileError::Unsupported.into()),
        _ => anyhow::bail!("{failure}"),
    }
}

fn profile_response_paths(
    profile: Option<proto::ProjectPath>,
    symbols: Option<proto::ProjectPath>,
) -> Result<[ProjectPath; 2]> {
    let profile = profile
        .and_then(ProjectPath::from_proto)
        .context("The host profile path is missing")?;
    let symbols = symbols
        .and_then(ProjectPath::from_proto)
        .context("The host symbols path is missing")?;
    ensure!(
        profile.worktree_id == symbols.worktree_id
            && profile.path.as_unix_str().ends_with(".json")
            && !profile.path.as_unix_str().ends_with(".syms.json")
            && profile.path.as_std_path().with_extension("syms.json") == symbols.path.as_std_path(),
        "The host returned an invalid Cargo profile recording and symbol pair"
    );
    Ok([profile, symbols])
}
