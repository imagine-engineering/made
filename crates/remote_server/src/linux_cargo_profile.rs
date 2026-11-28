use std::time::Duration;

use anyhow::{Context as _, Result};
use gpui::{App, AsyncApp, Context, Entity, Subscription, Task};
use project::{
    ProjectPath,
    cargo_workspace::{CargoTargetKind, CargoWorkspaceKey},
    cargo_workspace_store::HostCargoProfileSelection,
    trusted_worktrees::TrustedWorktrees,
};
use rpc::{AnyProtoClient, TypedEnvelope, proto};
use uuid::Uuid;

use crate::{HeadlessProject, NativeProjectTransport, headless_project::HostCargoProfileCapture};
use proto::{LinuxCargoProfileResponse as Response, ProfileResultStatus as Status};

pub(crate) fn register_handlers(session: &AnyProtoClient) {
    session.add_entity_request_handler(support);
    session.add_entity_request_handler(prepare);
    session.add_entity_request_handler(finish_review);
    session.add_entity_request_handler(cancel);
}

pub(crate) struct ProfileTransport {
    connection: Entity<NativeProjectTransport>,
    operation: Option<(Uuid, Phase)>,
    _subscription: Subscription,
}

enum Phase {
    Active {
        nonce: Uuid,
        cancel: async_channel::Sender<()>,
    },
    Review(Box<Review>),
}

struct Review {
    token: Uuid,
    capture: HostCargoProfileCapture,
    _expiry: Task<()>,
}

fn response(peer: proto::PeerId, operation_id: Vec<u8>, status: Status) -> Response {
    Response {
        status: status as i32,
        operation_id,
        native_receiver: peer.id,
        ..Default::default()
    }
}

fn operation_id(bytes: &[u8]) -> Result<Uuid, Status> {
    Uuid::from_slice(bytes)
        .ok()
        .filter(|id| !id.is_nil())
        .ok_or(Status::Error)
}

impl HeadlessProject {
    pub(crate) fn observe_linux_cargo_profile_transport(
        &mut self,
        connection: &Entity<NativeProjectTransport>,
        cx: &mut Context<Self>,
    ) {
        let subscription = cx.subscribe(connection, |host, _, _, _| {
            if let Some(transport) = host.linux_cargo_profile_transport.as_mut() {
                drop(transport.operation.take());
            }
        });
        self.linux_cargo_profile_transport = Some(ProfileTransport {
            connection: connection.clone(),
            operation: None,
            _subscription: subscription,
        });
    }

    fn profile_transport_for(
        &mut self,
        peer: proto::PeerId,
        forwarded: Option<proto::PeerId>,
        cx: &App,
    ) -> Result<&mut ProfileTransport, Status> {
        let transport = self
            .linux_cargo_profile_transport
            .as_mut()
            .ok_or(Status::Unsupported)?;
        if forwarded.is_some() || transport.connection.read(cx).connection_id != Some(peer) {
            return Err(Status::Restricted);
        }
        Ok(transport)
    }
}

async fn support(
    host: Entity<HeadlessProject>,
    envelope: TypedEnvelope<proto::GetLinuxCargoProfileSupport>,
    mut cx: AsyncApp,
) -> Result<Response> {
    let peer = envelope.sender_id;
    let status = host.update(&mut cx, |host, cx| {
        host.profile_transport_for(peer, envelope.original_sender_id, cx)?;
        crate::linux_cargo_runtime_observer_bytes().map_err(|_| Status::Unsupported)?;
        Ok(())
    });
    Ok(response(
        peer,
        Vec::new(),
        status.err().unwrap_or(Status::Ready),
    ))
}

async fn prepare(
    host: Entity<HeadlessProject>,
    envelope: TypedEnvelope<proto::PrepareLinuxCargoProfile>,
    mut cx: AsyncApp,
) -> Result<Response> {
    let peer = envelope.sender_id;
    let forwarded = envelope.original_sender_id;
    let request = envelope.payload;
    let id_bytes = request.operation_id.clone();
    let result = async {
        let id = operation_id(&id_bytes)?;
        let nonce = Uuid::new_v4();
        let (build, cancelled) = host.update(&mut cx, |host, cx| {
            let transport = host.profile_transport_for(peer, forwarded, cx)?;
            if transport.operation.is_some() {
                return Err(Status::Error);
            }
            if TrustedWorktrees::has_restricted_worktrees(&host.worktree_store, cx) {
                return Err(Status::Restricted);
            }
            crate::linux_cargo_runtime_observer_bytes().map_err(|_| Status::Unsupported)?;
            let selection = selection(request)?;
            let build = host.build_selected_host_cargo_profile(selection, cx).map_err(host_failure)?;
            let (cancel, cancelled) = async_channel::bounded(1);
            host.profile_transport_for(peer, forwarded, cx)?.operation = Some((id, Phase::Active { nonce, cancel }));
            Ok((build, cancelled))
        })?;
        let prepare = async {
            let workload = build.await?;
            let runtime = host.update(&mut cx, |host, cx| host.prepare_selected_host_cargo_profile_runtime(workload, cx))?.await?;
            host.update(&mut cx, |host, cx| host.prepare_selected_host_cargo_profile_capture(runtime, cx))?.await
        };
        let capture = await_active(prepare, cancelled).await;
        host.update(&mut cx, |host, cx| {
            let transport = host.profile_transport_for(peer, forwarded, cx)?;
            transport.retire_active(id, nonce)?;
            let capture = capture?;
            let [profile, symbols] = capture.paths().each_ref().map(ProjectPath::to_proto);
            let token = Uuid::new_v4();
            let expiry = cx.background_executor().timer(Duration::from_secs(300));
            let expiry = cx.spawn(async move |host, cx| {
                expiry.await;
                if let Err(error) = host.update(cx, |host, cx| {
                    if let Ok(transport) = host.profile_transport_for(peer, None, cx)
                        && matches!(transport.operation.as_ref(), Some((current, Phase::Review(review))) if *current == id && review.token == token)
                    {
                        drop(transport.operation.take());
                    }
                }) {
                    log::trace!("Linux Cargo profile review expired after its host closed: {error:#}");
                }
            });
            transport.operation = Some((id, Phase::Review(Box::new(Review {
                token,
                capture,
                _expiry: expiry,
            }))));
            Ok(Response {
                review_token: token.as_bytes().to_vec(),
                profile_path: Some(profile),
                symbols_path: Some(symbols),
                ..response(peer, id_bytes.clone(), Status::Ready)
            })
        })
    }.await;
    Ok(result.unwrap_or_else(|status| response(peer, id_bytes, status)))
}

fn selection(
    request: proto::PrepareLinuxCargoProfile,
) -> Result<HostCargoProfileSelection, Status> {
    let path = |path: Option<proto::ProjectPath>| {
        path.filter(|path| path.worktree_id != 0 && path.path.len() <= 4096)
            .and_then(ProjectPath::from_proto)
            .ok_or(Status::Error)
    };
    let target_kind = match request.target_kind() {
        proto::LinuxCargoProfileTargetKind::Binary => CargoTargetKind::Binary,
        proto::LinuxCargoProfileTargetKind::Example => CargoTargetKind::Example,
        _ => return Err(Status::Error),
    };
    Ok(HostCargoProfileSelection {
        workspace: CargoWorkspaceKey {
            root: path(request.workspace_root)?,
        },
        package_manifest: path(request.package_manifest)?,
        target_kind,
        target_name: request.target_name,
        preset_id: request.preset_id,
    })
}

async fn finish_review(
    host: Entity<HeadlessProject>,
    envelope: TypedEnvelope<proto::FinishLinuxCargoProfileReview>,
    mut cx: AsyncApp,
) -> Result<Response> {
    let peer = envelope.sender_id;
    let forwarded = envelope.original_sender_id;
    let request = envelope.payload;
    let result = async {
        let id = operation_id(&request.operation_id)?;
        let token = operation_id(&request.review_token)?;
        let nonce = Uuid::new_v4();
        let (capture, cancelled) = host.update(&mut cx, |host, cx| {
            let transport = host.profile_transport_for(peer, forwarded, cx)?;
            if !matches!(transport.operation.as_ref(), Some((operation, Phase::Review(review))) if *operation == id && review.token == token) {
                return Err(Status::Stale);
            }
            let Some((_, Phase::Review(review))) = transport.operation.take() else { return Err(Status::Stale) };
            if !request.approved {
                return Err(Status::Cancelled);
            }
            let capture = host.capture_selected_host_cargo_profile(review.capture, cx).map_err(host_failure)?;
            let (cancel, cancelled) = async_channel::bounded(1);
            host.profile_transport_for(peer, forwarded, cx)?.operation = Some((id, Phase::Active { nonce, cancel }));
            Ok((capture, cancelled))
        })?;
        let complete = async {
            let captured = capture.await?;
            let [profile, symbols] = captured.paths().each_ref().map(ProjectPath::to_proto);
            let task_run_id = captured.run_id().to_string();
            let registered = host.update(&mut cx, |host, cx| {
                let transport = host.profile_transport_for(peer, forwarded, cx)
                    .map_err(|_| anyhow::anyhow!("The native profiling connection changed"))?;
                anyhow::ensure!(transport.is_active(id, nonce), "The profile capture was cancelled");
                host.profile_results_store.update(cx, |store, cx| {
                    store.register_linux_task_profile(peer, captured, cx)
                })
            })?;
            registered.await.context("Registering the sealed Linux profile")?;
            Ok(Response {
                profile_path: Some(profile),
                symbols_path: Some(symbols),
                task_run_id,
                ..response(peer, request.operation_id.clone(), Status::Ready)
            })
        };
        let complete = await_active(complete, cancelled).await;
        host.update(&mut cx, |host, cx| {
            host.profile_transport_for(peer, forwarded, cx)?.retire_active(id, nonce)?;
            complete
        })
    }.await;
    Ok(result.unwrap_or_else(|status| response(peer, request.operation_id, status)))
}

async fn cancel(
    host: Entity<HeadlessProject>,
    envelope: TypedEnvelope<proto::CancelLinuxCargoProfile>,
    mut cx: AsyncApp,
) -> Result<Response> {
    let peer = envelope.sender_id;
    let request = envelope.payload;
    let result = host.update(&mut cx, |host, cx| {
        let id = operation_id(&request.operation_id)?;
        let transport = host.profile_transport_for(peer, envelope.original_sender_id, cx)?;
        if transport
            .operation
            .as_ref()
            .is_none_or(|(current, _)| *current != id)
        {
            return Err(Status::Stale);
        }
        drop(transport.operation.take());
        Ok(())
    });
    Ok(response(
        peer,
        request.operation_id,
        result.err().unwrap_or(Status::Cancelled),
    ))
}

impl ProfileTransport {
    fn is_active(&self, id: Uuid, nonce: Uuid) -> bool {
        matches!(self.operation.as_ref(), Some((current, Phase::Active { nonce: current_nonce, .. })) if *current == id && *current_nonce == nonce)
    }

    fn retire_active(&mut self, id: Uuid, nonce: Uuid) -> Result<(), Status> {
        if !self.is_active(id, nonce) {
            return Err(Status::Cancelled);
        }
        if let Some((_, Phase::Active { cancel, .. })) = self.operation.take() {
            cancel.close();
        }
        Ok(())
    }
}

async fn await_active<T>(
    task: impl Future<Output = Result<T>>,
    cancelled: async_channel::Receiver<()>,
) -> Result<T, Status> {
    use futures::future::{Either, select};
    match select(Box::pin(task), Box::pin(cancelled.recv())).await {
        Either::Left((result, _)) => result.map_err(host_failure),
        Either::Right(_) => Err(Status::Cancelled),
    }
}

fn host_failure(error: anyhow::Error) -> Status {
    log::warn!("Linux Cargo profiling request failed: {error:#}");
    Status::Error
}
