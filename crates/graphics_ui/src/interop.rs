use std::{fmt, sync::Arc};

use gpui::{App, AppContext as _, Context, Entity, Task, WeakEntity};
use graphics_core::evaluate::{
    CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgDocumentError, CpuSvgEncodedPngInput,
    CpuSvgEvaluationBudget,
};
use graphics_core::resources::{EncodedPngBudget, ImageResourceIdentity};
use sim_runtime::{
    AuthorizedCapabilities, OutputCommitReceipt, OutputOperationState,
    assets::{AssetIdentity, AssetNamespace, SharedAssetService},
};
use sim_tensor::CancellationToken;

use crate::{
    resources::ImageResourceError,
    session::{GraphiteArtboardSession, GraphiteEncodedPngLoad, GraphiteImageSession},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphiteAssetBindingError {
    Inactive,
    ProfileMismatch,
    InputNamespaceRequired,
    ServicePoisoned,
    MissingOutputAuthorization,
    InvalidOutputReceipt,
    Resource(ImageResourceError),
    Invalidation(Vec<ImageResourceError>),
}

impl fmt::Display for GraphiteAssetBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inactive => formatter.write_str("The Graphite asset binding is inactive."),
            Self::ProfileMismatch => formatter
                .write_str("The asset service, grant and request must use the same profile."),
            Self::InputNamespaceRequired => {
                formatter.write_str("Graphite image requests require an Input asset.")
            }
            Self::ServicePoisoned => formatter
                .write_str("The asset service is unavailable because its lock is poisoned."),
            Self::MissingOutputAuthorization => formatter
                .write_str("The Graphite asset binding has no supplied Output read authorization."),
            Self::InvalidOutputReceipt => formatter
                .write_str("The Output receipt does not match the committed image request."),
            Self::Resource(error) => error.fmt(formatter),
            Self::Invalidation(errors) => {
                formatter.write_str("Graphite asset invalidation failed:")?;
                for error in errors {
                    write!(formatter, " {error};")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for GraphiteAssetBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resource(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct GraphitePngAssetChoice {
    binding: WeakEntity<GraphiteAssetBinding>,
    record: sim_runtime::assets::AssetRecord,
}

impl GraphitePngAssetChoice {
    pub fn label(&self) -> String {
        self.record
            .identity
            .relative_path
            .to_string_lossy()
            .into_owned()
    }

    pub fn byte_size(&self) -> u64 {
        self.record.byte_size
    }

    pub(crate) fn binding(&self) -> anyhow::Result<Entity<GraphiteAssetBinding>> {
        self.binding
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("The PNG asset reader was closed. Reopen the chooser."))
    }

    pub(crate) fn request(&self) -> anyhow::Result<GraphiteAssetArtboardRequest> {
        use graphite_graph_craft::document::{
            NodeId,
            value::{DAffine2, DVec2, UVec2},
        };
        use graphite_std::{
            application_io::{ExportFormat, RenderConfig, TimingInformation},
            transform::Footprint,
            vector_types::vector::style::RenderMode,
        };
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_DOCUMENT_ID: AtomicU64 = AtomicU64::new(1);
        let document_id_encoding = NEXT_DOCUMENT_ID.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
            .map_err(|_| anyhow::anyhow!("The document identity limit was reached. Restart Sim before creating more documents."))?;
        let mut digest = [0_u8; 32];
        anyhow::ensure!(
            self.record.sha256.len() == 64,
            "The indexed PNG has an invalid content digest."
        );
        for (destination, encoded) in digest
            .iter_mut()
            .zip(self.record.sha256.as_bytes().as_chunks::<2>().0)
        {
            *destination = u8::from_str_radix(std::str::from_utf8(encoded)?, 16)?;
        }
        Ok(GraphiteAssetArtboardRequest {
            asset_identity: self.record.identity.clone(),
            identity: ImageResourceIdentity {
                asset_id: self.record.identity.to_reference()?.into(),
                // This labels a checked content snapshot; durable project versions
                // require the separately owned immutable asset contract.
                version_id: format!("sha256:{}", self.record.sha256).into(),
                content_sha256: digest,
            },
            placement: DAffine2::IDENTITY,
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 64 * 1024 * 1024,
                max_pixel_bytes: 128 * 1024 * 1024,
                max_metadata_bytes: 1024 * 1024,
            },
            document_id_encoding,
            node_id: NodeId(7001),
            parameters: CpuSvgArtboardParameters::default(),
            configuration: RenderConfig {
                viewport: Footprint {
                    transform: DAffine2::IDENTITY,
                    resolution: UVec2::new(1920, 1080),
                    quality: Footprint::DEFAULT.quality,
                },
                scale: 1.,
                export_format: ExportFormat::Svg,
                time: TimingInformation {
                    time: 0.,
                    animation_time: std::time::Duration::ZERO,
                },
                pointer: DVec2::ZERO,
                render_mode: RenderMode::Normal,
                for_export: true,
                for_eyedropper: false,
            },
            evaluation_budget: CpuSvgEvaluationBudget {
                max_viewport_pixels: 16 * 1024 * 1024,
                max_output_bytes: 64 * 1024 * 1024,
            },
        })
    }
}

pub struct GraphitePngAssetPage {
    pub choices: Vec<GraphitePngAssetChoice>,
    pub next_offset: Option<usize>,
}

pub(crate) fn query_authorized_pngs(
    offset: usize,
    text: Option<String>,
    cx: &mut App,
) -> anyhow::Result<Task<anyhow::Result<GraphitePngAssetPage>>> {
    use sim_runtime::assets::AssetQuery;
    let binding = crate::asset_binding(cx).ok_or_else(|| {
        anyhow::anyhow!(
            "The project asset reader is unavailable. Activate a native project profile and retry."
        )
    })?;
    anyhow::ensure!(
        binding.read(cx).active,
        "The project asset reader is inactive. Reopen the chooser."
    );
    let assets = binding.read(cx).assets.clone();
    let authorization = binding.read(cx).authorization.clone();
    let output_authorization = binding.read(cx).output_authorization.clone();
    let weak_binding = binding.downgrade();
    let background = cx.background_spawn({
        let assets = assets.clone();
        let authorization = authorization.clone();
        async move {
            let service = assets.lock().map_err(|_| {
                anyhow::anyhow!("The project asset reader is unavailable. Retry the request.")
            })?;
            Ok::<_, anyhow::Error>(service.list_authorized(
                &AssetQuery {
                    namespace: Some(AssetNamespace::Input),
                    text,
                    offset,
                    ..AssetQuery::default()
                },
                &authorization,
            )?)
        }
    });
    Ok(cx.spawn(async move |cx| {
        let page = background.await?;
        let current = weak_binding.read_with(cx, |binding, _cx| {
            binding.is_current_with_output_reader(
                &assets,
                &authorization,
                output_authorization.as_ref(),
            )
        })?;
        anyhow::ensure!(
            current,
            "The project asset profile changed. Reopen the chooser."
        );
        Ok(GraphitePngAssetPage {
            choices: page
                .records
                .into_iter()
                .filter(|record| {
                    record.content_type == "image/png"
                        && record.availability != sim_runtime::assets::AssetAvailability::Missing
                })
                .map(|record| GraphitePngAssetChoice {
                    binding: weak_binding.clone(),
                    record,
                })
                .collect(),
            next_offset: page.next_offset,
        })
    }))
}

#[derive(Clone)]
pub struct GraphitePublishedPngChoice {
    choice: GraphitePngAssetChoice,
    version: sim_runtime::ProjectAssetVersion,
}

impl GraphitePublishedPngChoice {
    pub fn label(&self) -> String {
        format!(
            "{} · {}",
            self.choice.label(),
            self.version.reference.version_id
        )
    }

    pub fn byte_size(&self) -> u64 {
        self.version.byte_size
    }

    pub(crate) fn binding(&self) -> anyhow::Result<Entity<GraphiteAssetBinding>> {
        self.choice.binding()
    }
}

pub struct GraphitePublishedPngPage {
    pub choices: Vec<GraphitePublishedPngChoice>,
    pub next_offset: Option<usize>,
}

pub(crate) fn query_published_png_versions(
    offset: usize,
    text: Option<String>,
    cx: &mut App,
) -> anyhow::Result<Task<anyhow::Result<GraphitePublishedPngPage>>> {
    let binding = crate::asset_binding(cx).ok_or_else(|| {
        anyhow::anyhow!(
            "The project reader is unavailable. Activate a native project profile and retry."
        )
    })?;
    let project = binding.update(cx, |binding, cx| binding.project_context(cx))?;
    let weak_binding = binding.downgrade();
    let background = cx.background_spawn({
        let project = project.clone();
        async move {
            let assets = project
                .assets
                .lock()
                .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
            let query = text.as_deref().unwrap_or_default().to_lowercase();
            let mut matches = Vec::new();
            for version in assets.project_png_versions(&project.output_reader)? {
                if version
                    .stored_identity
                    .relative_path
                    .starts_with(".sim/inputs")
                {
                    continue;
                }
                let Some(record) = assets.record(&version.stored_identity) else {
                    continue;
                };
                let choice = GraphitePublishedPngChoice {
                    choice: GraphitePngAssetChoice {
                        binding: weak_binding.clone(),
                        record,
                    },
                    version,
                };
                if choice.label().to_lowercase().contains(&query) {
                    matches.push(choice);
                }
            }
            let next_offset = offset.checked_add(64).filter(|next| *next < matches.len());
            Ok::<_, anyhow::Error>(GraphitePublishedPngPage {
                choices: matches.into_iter().skip(offset).take(64).collect(),
                next_offset,
            })
        }
    });
    Ok(cx.spawn(async move |cx| {
        let page = background.await?;
        anyhow::ensure!(
            cx.update(|cx| project.current(cx)),
            "The project profile changed. Reopen the chooser."
        );
        Ok(page)
    }))
}

fn verify_published_version(
    committer: &sim_runtime::OutputCommitter,
    version: &sim_runtime::ProjectAssetVersion,
) -> anyhow::Result<()> {
    if version
        .stored_identity
        .relative_path
        .starts_with(".sim/inputs")
    {
        anyhow::ensure!(
            version.dependencies.is_empty()
                && version.editable_source.is_none()
                && version.provenance == serde_json::json!({"kind":"retained-input"}),
            "The retained input version does not match its canonical source contract."
        );
        // Retained input bytes and catalog links outlive bounded publication receipts.
        return Ok(());
    }
    let operation = committer
        .operation(version.reference.version_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "The published PNG has no matching canonical publication. Reopen the chooser."
            )
        })?;
    anyhow::ensure!(
        operation.state == OutputOperationState::Committed
            && operation.identity == version.stored_identity
            && operation.sha256 == version.reference.content_sha256
            && operation.byte_size == version.byte_size
            && operation.proposal_id.is_some(),
        "The published PNG version does not match its canonical publication. Reopen the chooser."
    );
    let metadata: serde_json::Value = serde_json::from_slice(&operation.projection_metadata)
        .map_err(|_| {
            anyhow::anyhow!("The published PNG has no verified typed delivery metadata.")
        })?;
    anyhow::ensure!(
        version.original_identity == version.stored_identity
            && metadata.get("schema").and_then(serde_json::Value::as_str)
                == Some("sim.graphite.artboard-png-proposal.v1")
            && metadata == version.provenance
            && metadata.get("input_versions")
                == Some(&serde_json::to_value(&version.dependencies)?)
            && metadata.get("editable_source")
                == Some(&serde_json::to_value(&version.editable_source)?),
        "The published PNG provenance or dependency links do not match its canonical publication."
    );
    anyhow::ensure!(
        metadata.get("delivery_profile")
            == Some(&serde_json::json!({"format":"png","color_space":"sRGB","alpha":"straight"})),
        "This published PNG has no admitted sRGB/straight-alpha delivery profile. Its producer must declare a compatible image contract."
    );
    Ok(())
}

fn verify_published_closure(
    committer: &sim_runtime::OutputCommitter,
    versions: &[sim_runtime::ProjectAssetVersion],
    selected: &sim_runtime::ProjectAssetVersion,
    assets: &sim_runtime::assets::AssetService,
    reader: &AuthorizedCapabilities,
    cancellation: &CancellationToken,
) -> anyhow::Result<()> {
    let by_reference: std::collections::BTreeMap<_, _> = versions
        .iter()
        .map(|version| (&version.reference, version))
        .collect();
    let mut pending = vec![selected];
    let mut verified = std::collections::BTreeSet::new();
    while let Some(version) = pending.pop() {
        anyhow::ensure!(
            !cancellation.is_cancelled(),
            "Opening the published PNG was cancelled."
        );
        if !verified.insert(&version.reference) {
            continue;
        }
        verify_published_version(committer, version)?;
        if !version
            .stored_identity
            .relative_path
            .starts_with(".sim/inputs")
        {
            let source = version.editable_source.as_ref().ok_or_else(|| {
                anyhow::anyhow!("The published PNG has no verified editable producer document.")
            })?;
            let bytes = assets.read_verified(
                &source.identity,
                reader,
                cancellation,
                sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
            )?;
            anyhow::ensure!(
                encoded_sha256(&bytes) == source.content_sha256,
                "The published PNG producer document changed."
            );
            let document = sim_runtime::SimDocument::decode(&bytes)?;
            anyhow::ensure!(
                document.project_id.to_string() == reader.profile_id()
                    && document.documents.iter().any(|engine| {
                        engine.document_id == version.reference.asset_id
                            && engine.engine == "graphite"
                            && engine.source_revision
                                == graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION
                            && engine.serialization_format == "sim.graphite.legacy-artboard.v1"
                            && engine.dependencies == version.dependencies
                    }),
                "The published PNG asset identity does not match its canonical editable producer document."
            );
        }
        for dependency in &version.dependencies {
            pending.push(by_reference.get(dependency).copied().ok_or_else(|| {
                anyhow::anyhow!("The published PNG dependency version is missing: {dependency:?}")
            })?);
        }
    }
    Ok(())
}

pub struct GraphiteAssetImageRequest {
    pub asset_identity: AssetIdentity,
    pub identity: ImageResourceIdentity,
    pub transform: [f64; 6],
    pub encoded_budget: EncodedPngBudget,
    pub max_result_bytes: usize,
}

pub struct GraphiteAssetArtboardRequest {
    pub asset_identity: AssetIdentity,
    pub identity: ImageResourceIdentity,
    pub placement: graphite_graph_craft::document::value::DAffine2,
    pub encoded_budget: EncodedPngBudget,
    pub document_id_encoding: u64,
    pub node_id: graphite_graph_craft::document::NodeId,
    pub parameters: CpuSvgArtboardParameters,
    pub configuration: graphite_std::application_io::RenderConfig,
    pub evaluation_budget: CpuSvgEvaluationBudget,
}

#[derive(Clone, Debug)]
pub enum GraphiteArtboardOpenError {
    Read(Arc<crate::resources::GraphiteAssetReadError>),
    Document(CpuSvgDocumentError),
    Cancelled,
    BindingRevoked,
    Released,
}

impl fmt::Display for GraphiteArtboardOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => error.fmt(formatter),
            Self::Document(error) => error.fmt(formatter),
            Self::Cancelled => formatter.write_str("Artboard asset acquisition was cancelled"),
            Self::BindingRevoked => formatter.write_str("Artboard asset binding was revoked"),
            Self::Released => formatter.write_str("Artboard asset acquisition owner was released"),
        }
    }
}

impl std::error::Error for GraphiteArtboardOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error.as_ref()),
            Self::Document(error) => Some(error),
            _ => None,
        }
    }
}

struct PendingArtboardAcquisition {
    cancellation: Arc<CancellationToken>,
    task: Task<()>,
}

pub struct GraphiteArtboardAcquisition {
    task: Task<Result<Entity<GraphiteArtboardSession>, GraphiteArtboardOpenError>>,
    cancellation: Arc<CancellationToken>,
    completed: bool,
}

impl std::future::Future for GraphiteArtboardAcquisition {
    type Output = Result<Entity<GraphiteArtboardSession>, GraphiteArtboardOpenError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.cancellation.is_cancelled() {
            self.completed = true;
            return std::task::Poll::Ready(Err(GraphiteArtboardOpenError::Cancelled));
        }
        let outcome = std::pin::Pin::new(&mut self.task).poll(cx);
        if outcome.is_ready() {
            self.completed = true;
        }
        outcome
    }
}

struct BoundArtboardSession {
    session: WeakEntity<GraphiteArtboardSession>,
    cancellation: Arc<CancellationToken>,
}

impl Drop for GraphiteArtboardAcquisition {
    fn drop(&mut self) {
        if !self.completed {
            self.cancellation.cancel();
        }
    }
}

pub struct GraphiteAssetBinding {
    assets: SharedAssetService,
    authorization: AuthorizedCapabilities,
    output_authorization: Option<AuthorizedCapabilities>,
    publisher: Option<(sim_runtime::SharedOutputCommitter, AuthorizedCapabilities)>,
    active: bool,
    sessions: Vec<WeakEntity<GraphiteImageSession>>,
    artboard_sessions: Vec<BoundArtboardSession>,
    artboard_acquisitions: Vec<PendingArtboardAcquisition>,
    project_operations: Vec<std::sync::Weak<CancellationToken>>,
}

impl GraphiteAssetBinding {
    pub fn new(
        assets: SharedAssetService,
        authorization: AuthorizedCapabilities,
        cx: &mut App,
    ) -> Result<Entity<Self>, GraphiteAssetBindingError> {
        Self::new_with_output_reader(assets, authorization, None, cx)
    }

    pub fn new_with_output_reader(
        assets: SharedAssetService,
        input_authorization: AuthorizedCapabilities,
        output_authorization: Option<AuthorizedCapabilities>,
        cx: &mut App,
    ) -> Result<Entity<Self>, GraphiteAssetBindingError> {
        {
            let service = assets
                .lock()
                .map_err(|_| GraphiteAssetBindingError::ServicePoisoned)?;
            if service.roots().profile_id != input_authorization.profile_id()
                || output_authorization.as_ref().is_some_and(|authorization| {
                    authorization.profile_id() != service.roots().profile_id
                })
            {
                return Err(GraphiteAssetBindingError::ProfileMismatch);
            }
        }
        Ok(cx.new(|cx| {
            cx.on_release(|binding: &mut Self, cx| {
                if let Err(error) = binding.invalidate(cx) {
                    log::error!("Graphite asset binding release: {error}");
                }
            })
            .detach();
            Self {
                assets,
                authorization: input_authorization,
                output_authorization,
                publisher: None,
                active: true,
                sessions: Vec::new(),
                artboard_sessions: Vec::new(),
                artboard_acquisitions: Vec::new(),
                project_operations: Vec::new(),
            }
        }))
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn is_current(
        &self,
        assets: &SharedAssetService,
        authorization: &AuthorizedCapabilities,
    ) -> bool {
        self.is_current_with_output_reader(assets, authorization, None)
    }

    pub fn is_current_with_output_reader(
        &self,
        assets: &SharedAssetService,
        input_authorization: &AuthorizedCapabilities,
        output_authorization: Option<&AuthorizedCapabilities>,
    ) -> bool {
        self.active
            && Arc::ptr_eq(&self.assets, assets)
            && self.authorization == *input_authorization
            && self.output_authorization.as_ref() == output_authorization
    }

    pub fn open_artboard(
        &mut self,
        request: GraphiteAssetArtboardRequest,
        cx: &mut Context<Self>,
    ) -> Result<GraphiteArtboardAcquisition, GraphiteAssetBindingError> {
        if !self.active {
            return Err(GraphiteAssetBindingError::Inactive);
        }
        if request.asset_identity.profile_id != self.authorization.profile_id() {
            return Err(GraphiteAssetBindingError::ProfileMismatch);
        }
        if request.asset_identity.namespace != AssetNamespace::Input {
            return Err(GraphiteAssetBindingError::InputNamespaceRequired);
        }
        self.dispatch_artboard_request(request, self.authorization.clone(), cx)
    }

    pub fn open_committed_artboard(
        &mut self,
        request: GraphiteAssetArtboardRequest,
        receipt: &OutputCommitReceipt,
        cx: &mut Context<Self>,
    ) -> Result<GraphiteArtboardAcquisition, GraphiteAssetBindingError> {
        let authorization = self.committed_output_authorization(
            &request.asset_identity,
            &request.identity,
            receipt,
        )?;
        self.dispatch_artboard_request(request, authorization, cx)
    }

    fn dispatch_artboard_request(
        &mut self,
        request: GraphiteAssetArtboardRequest,
        authorization: AuthorizedCapabilities,
        cx: &mut Context<Self>,
    ) -> Result<GraphiteArtboardAcquisition, GraphiteAssetBindingError> {
        let cancellation = Arc::new(CancellationToken::default());
        let assets = self.assets.clone();
        let input_authorization = self.authorization.clone();
        let output_authorization = self.output_authorization.clone();
        let background = cx.background_spawn({
            let assets = assets.clone();
            let authorization = authorization.clone();
            let cancellation = cancellation.clone();
            async move {
                if cancellation.is_cancelled() {
                    return Err(GraphiteArtboardOpenError::Cancelled);
                }
                crate::resources::prepare_graphite_authorized_encoded_png_source(
                    &assets,
                    &request.asset_identity,
                    &authorization,
                    &cancellation,
                    &request.identity,
                    request.encoded_budget,
                )
                .map_err(|error| GraphiteArtboardOpenError::Read(Arc::new(error)))
            }
        });
        let (sender, receiver) = futures::channel::oneshot::channel();
        let task = cx.spawn({
            let cancellation = cancellation.clone();
            async move |binding, cx| {
                let source = background.await;
                let outcome = binding
                    .update(cx, |binding, cx| {
                        let Some(index) = binding
                            .artboard_acquisitions
                            .iter()
                            .position(|pending| Arc::ptr_eq(&pending.cancellation, &cancellation))
                        else {
                            return Err(GraphiteArtboardOpenError::BindingRevoked);
                        };
                        let pending = binding.artboard_acquisitions.swap_remove(index);
                        pending.task.detach();
                        if !binding.is_current_with_output_reader(
                            &assets,
                            &input_authorization,
                            output_authorization.as_ref(),
                        ) {
                            return Err(GraphiteArtboardOpenError::BindingRevoked);
                        }
                        if cancellation.is_cancelled() {
                            return Err(GraphiteArtboardOpenError::Cancelled);
                        }
                        let source = source?;
                        let input =
                            CpuSvgEncodedPngInput::new(source.source().clone(), request.placement)
                                .map_err(GraphiteArtboardOpenError::Document)?;
                        let document = CpuSvgArtboardDocument::new_with_encoded_png(
                            request.document_id_encoding,
                            request.node_id,
                            request.parameters,
                            request.configuration,
                            request.evaluation_budget,
                            input,
                        )
                        .map_err(GraphiteArtboardOpenError::Document)?;
                        let binding_reference = cx.entity().downgrade();
                        let session = cx.new({
                            let cancellation = cancellation.clone();
                            |cx| {
                                cx.on_release(move |_session: &mut GraphiteArtboardSession, cx| {
                                    cancellation.cancel();
                                    if let Err(error) = binding_reference.update(cx, |binding, _cx| {
                                        binding.artboard_sessions.retain(|bound| !Arc::ptr_eq(&bound.cancellation, &cancellation));
                                    }) {
                                        log::debug!("Artboard binding was released before its session: {error}");
                                    }
                                }).detach();
                                GraphiteArtboardSession::new(document)
                            }
                        });
                        binding.artboard_sessions.retain(|bound| bound.session.upgrade().is_some());
                        binding.artboard_sessions.push(BoundArtboardSession { session: session.downgrade(), cancellation: cancellation.clone() });
                        Ok(session)
                    })
                    .map_err(|_| {
                        // Weak-owner failure can precede GPUI's deferred release callback.
                        cancellation.cancel();
                        GraphiteArtboardOpenError::Released
                    })
                    .and_then(|outcome| outcome);
                if let Err(outcome) = sender.send(outcome) {
                    match outcome {
                        Ok(session) => {
                            session.update(cx, |session, cx| session.revoke_resource(cx))
                        }
                        Err(error) => {
                            log::debug!("Artboard asset acquisition receipt abandoned: {error}")
                        }
                    }
                }
            }
        });
        self.artboard_acquisitions.push(PendingArtboardAcquisition {
            cancellation: cancellation.clone(),
            task,
        });
        // GPUI can complete a Task before its caller polls it, so receipt ownership
        // must live outside the scheduled task and recheck cancellation on delivery.
        let task = cx.spawn(async move |_binding, _cx| {
            receiver
                .await
                .map_err(|_| GraphiteArtboardOpenError::Released)
                .and_then(|outcome| outcome)
        });
        Ok(GraphiteArtboardAcquisition {
            task,
            cancellation,
            completed: false,
        })
    }

    pub fn open_image(
        &mut self,
        request: GraphiteAssetImageRequest,
        cx: &mut Context<Self>,
    ) -> Result<Entity<GraphiteImageSession>, GraphiteAssetBindingError> {
        if !self.active {
            return Err(GraphiteAssetBindingError::Inactive);
        }
        if request.asset_identity.profile_id != self.authorization.profile_id() {
            return Err(GraphiteAssetBindingError::ProfileMismatch);
        }
        if request.asset_identity.namespace != AssetNamespace::Input {
            return Err(GraphiteAssetBindingError::InputNamespaceRequired);
        }
        self.dispatch_request(request, self.authorization.clone(), cx)
    }

    pub fn open_committed_image(
        &mut self,
        request: GraphiteAssetImageRequest,
        receipt: &OutputCommitReceipt,
        cx: &mut Context<Self>,
    ) -> Result<Entity<GraphiteImageSession>, GraphiteAssetBindingError> {
        let authorization = self.committed_output_authorization(
            &request.asset_identity,
            &request.identity,
            receipt,
        )?;
        self.dispatch_request(request, authorization, cx)
    }

    fn committed_output_authorization(
        &self,
        asset_identity: &AssetIdentity,
        identity: &ImageResourceIdentity,
        receipt: &OutputCommitReceipt,
    ) -> Result<AuthorizedCapabilities, GraphiteAssetBindingError> {
        if !self.active {
            return Err(GraphiteAssetBindingError::Inactive);
        }
        if asset_identity.profile_id != self.authorization.profile_id() {
            return Err(GraphiteAssetBindingError::ProfileMismatch);
        }
        let operation = receipt.operation();
        let receipt_reference = operation
            .identity
            .to_reference()
            .map_err(|_| GraphiteAssetBindingError::InvalidOutputReceipt)?;
        let requested_digest = identity
            .content_sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if operation.state != OutputOperationState::Committed
            || operation.identity.namespace != AssetNamespace::Output
            || asset_identity.namespace != AssetNamespace::Output
            || operation.identity != *asset_identity
            || identity.asset_id.as_ref() != receipt_reference
            || operation.sha256 != requested_digest
        {
            return Err(GraphiteAssetBindingError::InvalidOutputReceipt);
        }
        let authorization = self
            .output_authorization
            .clone()
            .ok_or(GraphiteAssetBindingError::MissingOutputAuthorization)?;
        Ok(authorization)
    }

    fn dispatch_request(
        &mut self,
        request: GraphiteAssetImageRequest,
        authorization: AuthorizedCapabilities,
        cx: &mut Context<Self>,
    ) -> Result<Entity<GraphiteImageSession>, GraphiteAssetBindingError> {
        let session = cx.new(|_| GraphiteImageSession::new(request.max_result_bytes));
        session
            .update(cx, |session, cx| {
                session.load_encoded_png(
                    GraphiteEncodedPngLoad {
                        assets: self.assets.clone(),
                        asset_identity: request.asset_identity,
                        authorization,
                        identity: request.identity,
                        transform: request.transform,
                        encoded_budget: request.encoded_budget,
                    },
                    cx,
                )
            })
            .map_err(GraphiteAssetBindingError::Resource)?;
        self.sessions.push(session.downgrade());
        Ok(session)
    }

    pub fn invalidate(&mut self, cx: &mut App) -> Result<(), GraphiteAssetBindingError> {
        self.active = false;
        for cancellation in self
            .project_operations
            .drain(..)
            .filter_map(|token| token.upgrade())
        {
            cancellation.cancel();
        }
        for pending in self.artboard_acquisitions.drain(..) {
            pending.cancellation.cancel();
            drop(pending);
        }
        for bound in self.artboard_sessions.drain(..) {
            bound.cancellation.cancel();
            if let Some(session) = bound.session.upgrade() {
                session.update(cx, |session, cx| session.revoke_resource(cx));
            }
        }
        let mut errors = Vec::new();
        for session in self.sessions.drain(..) {
            if let Some(session) = session.upgrade() {
                errors.extend(session.update(cx, |session, cx| session.revoke_all_resources(cx)));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphiteAssetBindingError::Invalidation(errors))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        resources::tests::{
            HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
        },
        session::ImageSessionDiagnostic,
    };
    use gpui::TestAppContext;
    use graphics_core::resources::CpuImageEdit;

    fn request(png: &IndexedRequestPng) -> GraphiteAssetImageRequest {
        GraphiteAssetImageRequest {
            asset_identity: png.asset_identity.clone(),
            identity: png.image_identity.clone(),
            transform: [1., 0., 0., 1., 0., 0.],
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
            max_result_bytes: 4096,
        }
    }

    fn binding(
        fixture: &HostReaderFixture,
        cx: &mut TestAppContext,
    ) -> Entity<GraphiteAssetBinding> {
        cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("genuine matching service and grant")
        })
    }

    #[gpui::test]
    fn at_graphite_asset_binding_real_input_retains_index_identity_and_existing_edit(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original encoded PNG");
        let before = fixture
            .assets
            .lock()
            .expect("service")
            .record(&png.asset_identity)
            .expect("indexed input");
        let binding = binding(&fixture, cx);
        let session = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("admitted request");
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            let state = session.resource_state();
            assert!(state.is_fresh());
            assert_eq!(state.revision(), 1);
            let image = state.committed().expect("real decoded image");
            assert_eq!(image.identity(), &png.image_identity);
            assert_eq!(image.descriptor(), &png.descriptor);
            assert!(session.diagnostic().is_none());
        });
        session.update(cx, |session, cx| {
            session.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        session.read_with(cx, |session, _| {
            assert!(session.result_is_fresh());
            assert_eq!(
                session.result().expect("original edit").operation(),
                CpuImageEdit::Invert
            );
        });
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("service")
                .record(&png.asset_identity),
            Some(before)
        );
        assert!(binding.read_with(cx, |binding, _| {
            binding.is_current(&fixture.assets, &fixture.input_authorization)
        }));
    }

    #[gpui::test]
    fn at_graphite_asset_binding_profile_namespace_and_initial_resource_fail_closed(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let foreign = sim_runtime::authorize_native_input_reader("foreign-profile")
            .expect("genuine foreign grant");
        cx.update(|cx| {
            assert!(matches!(
                GraphiteAssetBinding::new(fixture.assets.clone(), foreign, cx),
                Err(GraphiteAssetBindingError::ProfileMismatch)
            ));
        });
        let binding = binding(&fixture, cx);
        let mut foreign_request = request(&png);
        foreign_request.asset_identity.profile_id = "foreign-profile".into();
        assert!(matches!(
            binding.update(cx, |binding, cx| binding.open_image(foreign_request, cx)),
            Err(GraphiteAssetBindingError::ProfileMismatch)
        ));
        let mut output_request = request(&png);
        output_request.asset_identity.namespace = AssetNamespace::Output;
        assert!(matches!(
            binding.update(cx, |binding, cx| binding.open_image(output_request, cx)),
            Err(GraphiteAssetBindingError::InputNamespaceRequired)
        ));
        let mut malformed = request(&png);
        malformed.transform[0] = f64::NAN;
        assert!(matches!(
            binding.update(cx, |binding, cx| binding.open_image(malformed, cx)),
            Err(GraphiteAssetBindingError::Resource(
                ImageResourceError::NonFiniteTransform
            ))
        ));
        binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("no fabricated registration");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_denied_and_missing_are_existing_async_diagnostics(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let profile = fixture.input_authorization.profile_id();
        let capabilities = sim_runtime::CapabilitySet::new([]);
        let grant = sim_runtime::PermissionGrant::new(
            profile,
            "graphite-denied-fixture",
            capabilities.clone(),
            "isolated-component-test",
        )
        .expect("genuine empty grant");
        let denied = sim_runtime::PermissionPolicy::new(profile, [grant])
            .expect("actual policy")
            .authorize("graphite-denied-fixture", &capabilities)
            .expect("sealed empty authorization");
        let denied_binding = cx
            .update(|cx| GraphiteAssetBinding::new(fixture.assets.clone(), denied, cx))
            .expect("matching profile does not fabricate read access");
        let denied_session = denied_binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("read is asynchronous");
        cx.run_until_parked();
        denied_session.read_with(cx, |session, _| {
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
            assert!(!session.resource_state().has_active_request());
            assert!(!session.resource_state().is_fresh());
            assert_eq!(
                session.resource_state().diagnostic(),
                Some(&ImageResourceError::Denied)
            );
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    ImageResourceError::Denied
                ))
            );
        });
        let binding = binding(&fixture, cx);
        let mut missing = request(&png);
        missing.asset_identity = AssetIdentity::new(profile, AssetNamespace::Input, "missing.png")
            .expect("canonical missing reference");
        missing.identity.asset_id = missing
            .asset_identity
            .to_reference()
            .expect("missing path reference")
            .into();
        let expected_error = fixture
            .assets
            .lock()
            .expect("service")
            .read_verified(
                &missing.asset_identity,
                &fixture.input_authorization,
                &sim_tensor::CancellationToken::default(),
                16 * 1024,
            )
            .expect_err("the original service rejects an unindexed input");
        assert!(matches!(
            &expected_error,
            sim_runtime::assets::AssetError::UnknownAsset(identity)
                if identity == &missing.asset_identity
        ));
        let expected_failure = ImageResourceError::Failed(expected_error.to_string());
        let missing_session = binding
            .update(cx, |binding, cx| binding.open_image(missing, cx))
            .expect("asynchronous read");
        cx.run_until_parked();
        missing_session.read_with(cx, |session, _| {
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
            assert!(!session.resource_state().has_active_request());
            assert!(!session.resource_state().is_fresh());
            assert_eq!(
                session.resource_state().diagnostic(),
                Some(&expected_failure)
            );
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(expected_failure.clone()))
            );
        });
    }

    #[gpui::test]
    fn at_graphite_asset_binding_poisoned_service_does_not_create_a_lease(cx: &mut TestAppContext) {
        let fixture = host_reader_fixture().expect("canonical service");
        let assets = fixture.assets.clone();
        let poison = std::thread::spawn(move || {
            let _guard = assets.lock().expect("fixture service lock");
            panic!("deliberate poisoned-service fixture");
        });
        assert!(poison.join().is_err());
        cx.update(|cx| {
            assert!(matches!(
                GraphiteAssetBinding::new(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                ),
                Err(GraphiteAssetBindingError::ServicePoisoned)
            ));
        });
    }

    #[gpui::test]
    fn at_graphite_asset_binding_same_profile_service_and_grant_replacement_are_not_current(
        cx: &mut TestAppContext,
    ) {
        let first = host_reader_fixture().expect("first actual service");
        let second = host_reader_fixture().expect("same profile different actual service");
        let binding = binding(&first, cx);
        let capabilities = sim_runtime::CapabilitySet::new([sim_runtime::Capability::Asset {
            namespace: "input".into(),
            action: sim_runtime::AssetOperation::Read,
        }]);
        let grant = sim_runtime::PermissionGrant::new(
            first.input_authorization.profile_id(),
            "distinct-host-subject",
            capabilities.clone(),
            "isolated-component-test",
        )
        .expect("real alternate grant");
        let different =
            sim_runtime::PermissionPolicy::new(first.input_authorization.profile_id(), [grant])
                .expect("policy")
                .authorize("distinct-host-subject", &capabilities)
                .expect("sealed alternative");
        binding.read_with(cx, |binding, _| {
            assert!(binding.is_current(&first.assets, &first.input_authorization));
            assert!(!binding.is_current(&second.assets, &first.input_authorization));
            assert!(!binding.is_current(&first.assets, &different));
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("empty invalidation");
        assert!(!binding.read_with(cx, |binding, _| {
            binding.is_current(&first.assets, &first.input_authorization)
        }));
    }

    #[gpui::test]
    fn at_graphite_asset_binding_invalidation_cancels_pending_and_revokes_accepted_edit(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let binding = binding(&fixture, cx);
        let accepted = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("accepted request");
        cx.run_until_parked();
        accepted.update(cx, |session, cx| {
            session.set_operation(Some(CpuImageEdit::MakeOpaque), cx)
        });
        cx.run_until_parked();
        assert!(accepted.read_with(cx, |session, _| session.result_is_fresh()));
        let pending = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("pending request");
        assert!(pending.read_with(cx, |session, _| {
            session.resource_state().has_active_request()
        }));
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("all live owners revoked");
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        assert!(matches!(
            binding.update(cx, |binding, cx| binding.open_image(request(&png), cx)),
            Err(GraphiteAssetBindingError::Inactive)
        ));
        cx.run_until_parked();
        accepted.read_with(cx, |session, _| {
            assert!(session.resource_state().is_revoked());
            assert!(!session.result_is_fresh());
            assert!(session.result().is_none());
            assert_eq!(session.resource_state().revision(), 1);
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    ImageResourceError::Revoked
                ))
            );
        });
        pending.read_with(cx, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    ImageResourceError::Revoked
                ))
            );
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("repeat has no registrations");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_invalidation_revokes_current_and_pending_successor_versions(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let binding = binding(&fixture, cx);
        let changed = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("first session");
        let pending = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("pending successor session");
        let other = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("second session");
        cx.run_until_parked();
        let mut next = request(&png);
        next.identity.version_id = "explicit-test-successor".into();
        let next_identity = next.identity.clone();
        changed
            .update(cx, |session, cx| {
                session.load_encoded_png(
                    GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: next.asset_identity,
                        authorization: fixture.input_authorization.clone(),
                        identity: next.identity,
                        transform: next.transform,
                        encoded_budget: next.encoded_budget,
                    },
                    cx,
                )
            })
            .expect("explicit independently supplied replacement");
        cx.run_until_parked();
        changed.update(cx, |session, cx| {
            session.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        changed.read_with(cx, |session, _| {
            assert!(session.resource_state().is_fresh());
            assert_eq!(session.resource_state().revision(), 2);
            assert_eq!(
                session
                    .resource_state()
                    .committed()
                    .expect("successor")
                    .identity(),
                &next_identity
            );
            assert!(session.result_is_fresh());
            assert_eq!(
                session.result().expect("successor edit").input_identity(),
                &next_identity
            );
        });
        let mut pending_request = request(&png);
        pending_request.identity.version_id = "explicit-pending-successor".into();
        let pending_identity = pending_request.identity.clone();
        pending
            .update(cx, |session, cx| {
                session.load_encoded_png(
                    GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: pending_request.asset_identity,
                        authorization: fixture.input_authorization.clone(),
                        identity: pending_request.identity,
                        transform: pending_request.transform,
                        encoded_budget: pending_request.encoded_budget,
                    },
                    cx,
                )
            })
            .expect("pending explicit successor");
        let pending_original = pending.read_with(cx, |session, _| {
            assert_eq!(
                session
                    .resource_state()
                    .active_encoded_request()
                    .expect("pending")
                    .identity(),
                &pending_identity
            );
            assert_eq!(session.resource_state().revision(), 1);
            session
                .resource_state()
                .committed()
                .expect("prior image")
                .clone()
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("valid version refresh is revoked without a stale-tracker error");
        assert!(!binding.read_with(cx, |binding, _| binding.is_active()));
        for session in [&changed, &pending, &other] {
            session.read_with(cx, |session, _| {
                assert!(session.resource_state().is_revoked());
                assert!(!session.resource_state().is_fresh());
                assert!(!session.resource_state().has_active_request());
                assert!(!session.result_is_fresh());
                assert!(session.result().is_none());
                assert_eq!(
                    session.diagnostic(),
                    Some(&ImageSessionDiagnostic::Resource(
                        ImageResourceError::Revoked
                    ))
                );
            });
        }
        cx.run_until_parked();
        pending.read_with(cx, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(session.resource_state().is_revoked());
            assert_eq!(session.resource_state().revision(), 1);
            assert!(Arc::ptr_eq(
                session
                    .resource_state()
                    .committed()
                    .expect("last revoked image"),
                &pending_original
            ));
            assert_eq!(
                session
                    .resource_state()
                    .committed()
                    .expect("last image")
                    .identity(),
                &png.image_identity
            );
            assert!(session.result().is_none());
        });
        changed.read_with(cx, |session, _| {
            assert_eq!(session.resource_state().revision(), 2);
            assert_eq!(
                session
                    .resource_state()
                    .committed()
                    .expect("revoked successor")
                    .identity(),
                &next_identity
            );
            assert!(session.resource_state().is_revoked());
            assert!(session.result().is_none());
        });
        binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("repeat has no registrations");
    }

    #[gpui::test]
    fn at_graphite_asset_binding_release_cancels_live_pending_and_ignores_released_session(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("original PNG");
        let binding = binding(&fixture, cx);
        let released = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("released session");
        let weak_session = released.downgrade();
        drop(released);
        let pending = binding
            .update(cx, |binding, cx| binding.open_image(request(&png), cx))
            .expect("live pending session");
        pending.read_with(cx, |session, _| {
            assert!(session.resource_state().has_active_request());
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
        });
        let weak_binding = binding.downgrade();
        drop(binding);
        cx.update(|_| {});
        assert!(weak_binding.upgrade().is_none());
        assert!(weak_session.upgrade().is_none());
        pending.read_with(cx, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    ImageResourceError::Revoked
                ))
            );
        });
        cx.run_until_parked();
        assert!(weak_binding.upgrade().is_none());
        assert!(weak_session.upgrade().is_none());
        pending.read_with(cx, |session, _| {
            assert!(!session.resource_state().has_active_request());
            assert!(session.resource_state().committed().is_none());
            assert_eq!(session.resource_state().revision(), 0);
            assert_eq!(
                session.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    ImageResourceError::Revoked
                ))
            );
        });
    }

    #[cfg(test)]
    pub(crate) mod output_reader_tests {
        use super::*;
        use sha2::{Digest as _, Sha256};
        use sim_runtime::{
            AssetOperation, Capability, CapabilitySet, OutputCommitter, OutputProposal,
        };
        use sim_tensor::CancellationToken;

        fn grant(
            fixture: &HostReaderFixture,
            capabilities: CapabilitySet,
        ) -> AuthorizedCapabilities {
            let profile = fixture.input_authorization.profile_id();
            let grant = sim_runtime::PermissionGrant::new(
                profile,
                "graphite-output-fixture",
                capabilities.clone(),
                "isolated-output-component-test",
            )
            .expect("real permission grant");
            sim_runtime::PermissionPolicy::new(profile, [grant])
                .expect("actual policy")
                .authorize("graphite-output-fixture", &capabilities)
                .expect("sealed supplied authorization")
        }

        pub(crate) fn published(
            fixture: &HostReaderFixture,
            png: &IndexedRequestPng,
            proposal_id: &str,
            prefix: &str,
        ) -> (
            OutputCommitReceipt,
            GraphiteAssetImageRequest,
            AuthorizedCapabilities,
            AuthorizedCapabilities,
        ) {
            let writer = grant(
                fixture,
                CapabilitySet::new([AssetNamespace::Output, AssetNamespace::Temporary].map(
                    |namespace| Capability::Asset {
                        namespace: namespace.locator_type().to_owned(),
                        action: AssetOperation::Write,
                    },
                )),
            );
            let reader = grant(
                fixture,
                CapabilitySet::new([Capability::Asset {
                    namespace: AssetNamespace::Output.locator_type().to_owned(),
                    action: AssetOperation::Read,
                }]),
            );
            let roots = fixture.assets.lock().expect("service").roots().clone();
            let mut publisher = OutputCommitter::open(roots).expect("actual original committer");
            let proposal = OutputProposal::new(
                proposal_id.parse().expect("fixture proposal UUID"),
                AssetNamespace::Output,
                prefix,
                "png",
                0,
                png.descriptor.width,
                png.descriptor.height,
                png.encoded.clone(),
            )
            .expect("real PNG proposal");
            let receipts = publisher
                .commit_proposal_batch_and_register(
                    &[proposal],
                    "2026-10-08T12:00:00+00:00"
                        .parse()
                        .expect("fixed fixture timestamp"),
                    &mut fixture.assets.lock().expect("same actual service"),
                    &writer,
                    &CancellationToken::default(),
                )
                .expect("actual combined publication and index transaction");
            assert_eq!(receipts.len(), 1);
            let receipt = receipts.into_iter().next().expect("published receipt");
            assert_eq!(receipt.operation().state, OutputOperationState::Committed);
            let mut output = request(png);
            output.asset_identity = receipt.operation().identity.clone();
            output.identity.asset_id = output
                .asset_identity
                .to_reference()
                .expect("real Output reference")
                .into();
            output.identity.version_id = "explicit-output-test-version".into();
            output.identity.content_sha256 = Sha256::digest(&png.encoded).into();
            (receipt, output, writer, reader)
        }

        #[gpui::test]
        fn at_graphite_output_receipt_reader_decodes_edits_and_revokes_without_mutating_inputs(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("service");
            let png = indexed_request_png(&fixture, 4).expect("actual PNG producer");
            let input_before = fixture
                .assets
                .lock()
                .expect("service")
                .record(&png.asset_identity)
                .expect("Input indexed record");
            let (receipt, output, _, reader) = published(
                &fixture,
                &png,
                "10000000-0000-0000-0000-000000000001",
                "graphite-output",
            );
            let receipt_before = receipt.clone();
            let output_identity = output.asset_identity.clone();
            let image_identity = output.identity.clone();
            let output_before = fixture
                .assets
                .lock()
                .expect("service")
                .record(&output_identity)
                .expect("published index");
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader.clone()),
                        cx,
                    )
                })
                .expect("separately supplied output reader");
            binding.read_with(cx, |binding, _| {
                assert!(binding.is_current_with_output_reader(
                    &fixture.assets,
                    &fixture.input_authorization,
                    Some(&reader)
                ));
                assert!(!binding.is_current(&fixture.assets, &fixture.input_authorization));
                assert!(!binding.is_current_with_output_reader(
                    &fixture.assets,
                    &fixture.input_authorization,
                    Some(&fixture.input_authorization)
                ));
            });
            let input_session = binding
                .update(cx, |binding, cx| binding.open_image(request(&png), cx))
                .expect("the original Input-only path remains available");
            let mut rejected_generic = request(&png);
            rejected_generic.asset_identity = output_identity.clone();
            assert!(matches!(
                binding.update(cx, |binding, cx| binding.open_image(rejected_generic, cx)),
                Err(GraphiteAssetBindingError::InputNamespaceRequired)
            ));
            let session = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_image(output, &receipt, cx)
                })
                .expect("genuine committed Output request");
            cx.run_until_parked();
            input_session.read_with(cx, |session, _| {
                assert!(session.resource_state().is_fresh());
                assert_eq!(
                    session
                        .resource_state()
                        .committed()
                        .expect("Input image")
                        .identity(),
                    &png.image_identity
                );
            });
            let source = session.read_with(cx, |session, _| {
                assert!(session.resource_state().is_fresh());
                assert_eq!(session.resource_state().revision(), 1);
                let image = session
                    .resource_state()
                    .committed()
                    .expect("original PNG decoder");
                assert_eq!(image.identity(), &image_identity);
                assert_eq!(image.descriptor(), &png.descriptor);
                image.clone()
            });
            session.update(cx, |session, cx| {
                session.set_operation(Some(CpuImageEdit::Invert), cx)
            });
            cx.run_until_parked();
            session.read_with(cx, |session, _| {
                assert!(session.result_is_fresh());
                assert_eq!(
                    session.result().expect("edit").image(),
                    graphics_core::resources::evaluate_cpu_image_edit(
                        &source,
                        CpuImageEdit::Invert,
                        4096
                    )
                    .expect("original node")
                    .image()
                );
                assert!(Arc::ptr_eq(
                    session
                        .resource_state()
                        .committed()
                        .expect("unchanged source"),
                    &source
                ));
            });
            let mut pending_request = request(&png);
            pending_request.asset_identity = output_identity.clone();
            pending_request.identity = image_identity.clone();
            let pending = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_image(pending_request, &receipt, cx)
                })
                .expect("pending output decode");
            assert!(pending.read_with(cx, |session, _| {
                session.resource_state().has_active_request()
            }));
            binding
                .update(cx, |binding, cx| binding.invalidate(cx))
                .expect("all Input and Output leases invalidated");
            cx.run_until_parked();
            session.read_with(cx, |session, _| {
                assert!(session.resource_state().is_revoked());
                assert!(!session.resource_state().is_fresh());
                assert!(session.result().is_none());
                assert_eq!(session.resource_state().revision(), 1);
            });
            input_session.read_with(cx, |session, _| {
                assert!(session.resource_state().is_revoked());
                assert!(!session.resource_state().is_fresh());
            });
            pending.read_with(cx, |session, _| {
                assert!(!session.resource_state().has_active_request());
                assert!(session.resource_state().committed().is_none());
                assert_eq!(session.resource_state().revision(), 0);
            });
            let service = fixture.assets.lock().expect("same service");
            assert_eq!(
                service.record(&png.asset_identity).expect("Input retained"),
                input_before
            );
            assert_eq!(
                service.record(&output_identity).expect("Output retained"),
                output_before
            );
            assert_eq!(
                service
                    .read_verified(
                        &output_identity,
                        &reader,
                        &CancellationToken::default(),
                        16 * 1024
                    )
                    .expect("same producer bytes"),
                png.encoded
            );
            assert_eq!(receipt, receipt_before);
        }

        #[gpui::test]
        fn at_graphite_output_receipt_reader_requires_separate_matching_grant(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("service");
            let png = indexed_request_png(&fixture, 3).expect("actual producer");
            let (receipt, output, _, _) = published(
                &fixture,
                &png,
                "20000000-0000-0000-0000-000000000001",
                "output-grant",
            );
            let input_binding = binding(&fixture, cx);
            assert!(matches!(
                input_binding.update(cx, |binding, cx| binding
                    .open_committed_image(output, &receipt, cx)),
                Err(GraphiteAssetBindingError::MissingOutputAuthorization)
            ));
            input_binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
            let wrongly_supplied = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(fixture.input_authorization.clone()),
                        cx,
                    )
                })
                .expect("same profile is not permission escalation");
            let mut output = request(&png);
            output.asset_identity = receipt.operation().identity.clone();
            output.identity.asset_id = output
                .asset_identity
                .to_reference()
                .expect("Output reference")
                .into();
            let denied = wrongly_supplied
                .update(cx, |binding, cx| {
                    binding.open_committed_image(output, &receipt, cx)
                })
                .expect("actual service decides permission asynchronously");
            cx.run_until_parked();
            denied.read_with(cx, |session, _| {
                assert_eq!(
                    session.diagnostic(),
                    Some(&ImageSessionDiagnostic::Resource(
                        ImageResourceError::Denied
                    ))
                );
                assert!(!session.resource_state().has_active_request());
                assert!(session.resource_state().committed().is_none());
                assert_eq!(session.resource_state().revision(), 0);
            });
            let capabilities = CapabilitySet::new([Capability::Asset {
                namespace: "output".to_owned(),
                action: AssetOperation::Read,
            }]);
            let foreign = sim_runtime::PermissionGrant::new(
                "foreign-output-profile",
                "reader",
                capabilities.clone(),
                "isolated-component-test",
            )
            .expect("actual foreign grant");
            let foreign = sim_runtime::PermissionPolicy::new("foreign-output-profile", [foreign])
                .expect("actual policy")
                .authorize("reader", &capabilities)
                .expect("sealed foreign reader");
            assert!(matches!(
                cx.update(|cx| GraphiteAssetBinding::new_with_output_reader(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    Some(foreign),
                    cx
                )),
                Err(GraphiteAssetBindingError::ProfileMismatch)
            ));
        }

        #[gpui::test]
        fn at_graphite_output_receipt_reader_rejects_namespace_identity_profile_and_digest_mismatch(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("service");
            let png = indexed_request_png(&fixture, 4).expect("actual producer");
            let (receipt, output, _, reader) = published(
                &fixture,
                &png,
                "30000000-0000-0000-0000-000000000001",
                "output-identity",
            );
            let before = receipt.clone();
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader),
                        cx,
                    )
                })
                .expect("separate grant");
            let mut wrong_namespace = request(&png);
            wrong_namespace.asset_identity = output.asset_identity.clone();
            wrong_namespace.asset_identity.namespace = AssetNamespace::Input;
            assert!(matches!(
                binding.update(cx, |binding, cx| binding.open_committed_image(
                    wrong_namespace,
                    &receipt,
                    cx
                )),
                Err(GraphiteAssetBindingError::InvalidOutputReceipt)
            ));
            let mut wrong_path = request(&png);
            wrong_path.asset_identity = AssetIdentity::new(
                fixture.input_authorization.profile_id(),
                AssetNamespace::Output,
                "uncommitted.png",
            )
            .expect("canonical wrong reference");
            assert!(matches!(
                binding.update(cx, |binding, cx| binding
                    .open_committed_image(wrong_path, &receipt, cx)),
                Err(GraphiteAssetBindingError::InvalidOutputReceipt)
            ));
            let mut wrong_profile = request(&png);
            wrong_profile.asset_identity = output.asset_identity.clone();
            wrong_profile.asset_identity.profile_id = "foreign-output-profile".into();
            assert!(matches!(
                binding.update(cx, |binding, cx| binding.open_committed_image(
                    wrong_profile,
                    &receipt,
                    cx
                )),
                Err(GraphiteAssetBindingError::ProfileMismatch)
            ));
            let mut wrong_digest = output;
            wrong_digest.identity.content_sha256[0] ^= 1;
            assert!(matches!(
                binding.update(cx, |binding, cx| binding.open_committed_image(
                    wrong_digest,
                    &receipt,
                    cx
                )),
                Err(GraphiteAssetBindingError::InvalidOutputReceipt)
            ));
            binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
            assert_eq!(receipt, before);
        }

        #[gpui::test]
        fn at_graphite_output_receipt_reader_rejects_changed_indexed_bytes_and_recovers_from_new_receipt(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("service");
            let png = indexed_request_png(&fixture, 4).expect("actual RGBA producer");
            let (receipt, output, writer, reader) = published(
                &fixture,
                &png,
                "40000000-0000-0000-0000-000000000001",
                "output-stale",
            );
            let before = receipt.clone();
            let replacement =
                indexed_request_png(&fixture, 3).expect("actual different RGB producer");
            assert_ne!(png.encoded, replacement.encoded);
            fixture
                .assets
                .lock()
                .expect("service")
                .write_exact(
                    &output.asset_identity,
                    &replacement.encoded,
                    Default::default(),
                    sim_runtime::assets::AssetCollisionPolicy::Replace,
                    &writer,
                    &CancellationToken::default(),
                )
                .expect("actual indexed Output replacement");
            let expected = graphics_core::resources::admit_encoded_png(
                output.identity.clone(),
                output.transform,
                Arc::from(replacement.encoded.as_slice()),
                output.encoded_budget,
            )
            .expect_err("old receipt digest cannot admit changed PNG");
            assert!(matches!(
                expected,
                graphics_core::resources::PngAdmissionError::Image(
                    graphics_core::resources::ImageAdmissionError::DigestMismatch
                )
            ));
            let expected_diagnostic = expected.to_string();
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader),
                        cx,
                    )
                })
                .expect("supplied reader");
            let stale = binding.update(cx, |binding, cx| binding.open_committed_image(output, &receipt, cx)).expect("receipt only validates original commit, actual reader must check current bytes");
            cx.run_until_parked();
            stale.read_with(cx, |session, _| {
                assert_eq!(
                    session.diagnostic(),
                    Some(&ImageSessionDiagnostic::Resource(
                        ImageResourceError::Failed(expected_diagnostic.clone())
                    ))
                );
                assert!(session.resource_state().committed().is_none());
                assert_eq!(session.resource_state().revision(), 0);
                assert!(!session.resource_state().has_active_request());
            });
            let (successor, mut next, _, _) = published(
                &fixture,
                &replacement,
                "40000000-0000-0000-0000-000000000002",
                "output-successor",
            );
            next.identity.version_id = "explicit-new-output-test-version".into();
            let next_identity = next.identity.clone();
            let fresh = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_image(next, &successor, cx)
                })
                .expect("genuine successor commit");
            cx.run_until_parked();
            fresh.read_with(cx, |session, _| {
                assert!(session.resource_state().is_fresh());
                assert_eq!(session.resource_state().revision(), 1);
                assert_eq!(
                    session
                        .resource_state()
                        .committed()
                        .expect("new original decode")
                        .identity(),
                    &next_identity
                );
                assert_eq!(
                    session
                        .resource_state()
                        .committed()
                        .expect("new image")
                        .descriptor(),
                    &replacement.descriptor
                );
            });
            assert_eq!(receipt, before);
            assert_eq!(successor.operation().state, OutputOperationState::Committed);
        }

        #[gpui::test]
        fn at_graphite_output_receipt_reader_rejects_relabeling_before_session_registration(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("real service");
            let png = indexed_request_png(&fixture, 4).expect("actual PNG producer");
            let (receipt, mut output, _, reader) = published(
                &fixture,
                &png,
                "50000000-0000-0000-0000-000000000001",
                "output-relabel",
            );
            let original_receipt = receipt.clone();
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader),
                        cx,
                    )
                })
                .expect("supplied Output reader");
            let canonical_asset_id = output.identity.asset_id.clone();
            output.identity.asset_id = png.image_identity.asset_id.clone();
            assert_ne!(output.identity.asset_id, canonical_asset_id);
            assert_eq!(output.asset_identity, receipt.operation().identity);
            let original_digest: [u8; 32] = Sha256::digest(&png.encoded).into();
            assert_eq!(output.identity.content_sha256, original_digest);
            assert!(matches!(
                binding.update(cx, |binding, cx| binding
                    .open_committed_image(output, &receipt, cx)),
                Err(GraphiteAssetBindingError::InvalidOutputReceipt)
            ));
            binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
            cx.run_until_parked();
            binding.read_with(cx, |binding, _| assert!(binding.sessions.is_empty()));
            assert_eq!(receipt, original_receipt);
        }

        fn committed_artboard_request(
            png: &IndexedRequestPng,
            receipt: &OutputCommitReceipt,
        ) -> GraphiteAssetArtboardRequest {
            let mut request = crate::interop::artboard_binding_controls::request(png);
            request.asset_identity = receipt.operation().identity.clone();
            request.identity.asset_id = request
                .asset_identity
                .to_reference()
                .expect("Output reference")
                .into();
            request.identity.version_id = "supplied-committed-artboard-label".into();
            request.placement = graphite_glam::DAffine2::from_scale_angle_translation(
                graphite_glam::DVec2::new(12., 4.),
                0.,
                graphite_glam::DVec2::new(3., 2.),
            );
            request
        }

        #[gpui::test]
        async fn at_r08_r28_committed_output_artboard_edits_preview_and_png_retain_genuine_source(
            cx: &mut TestAppContext,
        ) {
            use graphite_glam::{DAffine2, DVec2};
            let fixture = host_reader_fixture().expect("real host service");
            let png = indexed_request_png(&fixture, 4).expect("actual tensor PNG producer");
            let (receipt, _, _, reader) = published(
                &fixture,
                &png,
                "61000000-0000-0000-0000-000000000001",
                "committed-artboard",
            );
            let receipt_before = receipt.clone();
            let output_identity = receipt.operation().identity.clone();
            let output_before = fixture
                .assets
                .lock()
                .expect("service")
                .record(&output_identity)
                .expect("Output record");
            let input_before = fixture
                .assets
                .lock()
                .expect("service")
                .record(&png.asset_identity)
                .expect("Input record");
            let request = committed_artboard_request(&png, &receipt);
            let identity = request.identity.clone();
            let configuration = request.configuration;
            let placement = request.placement;
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader.clone()),
                        cx,
                    )
                })
                .expect("genuine supplied Output reader");
            let session = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_artboard(request, &receipt, cx)
                })
                .expect("committed admission")
                .await
                .expect("authenticated source seal");
            let source = session.read_with(cx, |session, _| {
                let input = session
                    .document()
                    .encoded_png_input()
                    .expect("retained source");
                assert_eq!(input.source().identity(), &identity);
                assert_eq!(
                    (input.source().width(), input.source().height()),
                    (png.descriptor.width, png.descriptor.height)
                );
                assert_eq!(
                    input.placement().to_cols_array().map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert_eq!(session.configuration(), configuration);
                assert!(!session.resource_is_revoked());
                input.source().clone()
            });
            let original_network =
                session.read_with(cx, |session, _| session.document().network().clone());
            let original = session
                .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
                .expect("original request")
                .await
                .expect("original network and native conversion");
            let original_output = original.output().output().clone();
            let original_frame = original
                .image()
                .as_bytes(0)
                .expect("single actual BGRA frame")
                .to_vec();
            assert_eq!(original_frame.len(), 31 * 17 * 4);
            assert_eq!(
                original_frame.get(original_frame.len() - 4..),
                Some(&[255, 255, 255, 255][..])
            );
            assert!(
                original_frame
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|pixel| pixel[2] > pixel[0] && pixel[2] > pixel[1])
            );
            assert_eq!(original_output.metadata.backgrounds.len(), 1);
            assert_eq!(
                original_output.metadata.backgrounds[0].dimensions,
                DVec2::new(31., 17.)
            );
            assert!(
                original_output
                    .metadata
                    .clip_targets
                    .contains(&graphite_graph_craft::document::NodeId(7142))
            );
            let mut parameters = session.read_with(cx, |session, _| session.parameters());
            parameters.background = graphite_std::Color::BLUE;
            session.update(cx, |session, cx| {
                assert!(
                    session
                        .apply(session.revision(), parameters, configuration, cx)
                        .expect("canonical parameter edit")
                );
                assert!(
                    session
                        .apply_image_placement(
                            session.revision(),
                            DAffine2::from_scale_angle_translation(
                                DVec2::new(12., 4.),
                                0.,
                                DVec2::new(7., 5.)
                            ),
                            cx
                        )
                        .expect("canonical placement edit")
                );
                assert!(!session.preview_is_current(&original));
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("same source")
                        .source(),
                    &source
                ));
            });
            let edited = session
                .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
                .expect("edited request")
                .await
                .expect("genuine edited output");
            let edited_frame = edited.image().as_bytes(0).expect("edited frame");
            assert_ne!(edited_frame, original_frame);
            assert_eq!(
                edited_frame.get(edited_frame.len() - 4..),
                Some(&[255, 0, 0, 255][..])
            );
            let prepared = session
                .update(cx, |session, cx| {
                    session.prepare_png(edited.clone(), 31 * 17 * 4, 64 * 1024, cx)
                })
                .expect("native PNG request")
                .await
                .expect("shared byte encoder");
            let expected_rgba = edited_frame
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
                .collect::<Vec<_>>();
            let decoded = image::load_from_memory(prepared.png().encoded_bytes())
                .expect("actual PNG")
                .into_rgba8();
            assert_eq!(decoded.dimensions(), (31, 17));
            assert_eq!(decoded.as_raw(), &expected_rgba);
            assert_eq!(
                prepared.pixels_sha256(),
                <[u8; 32]>::from(Sha256::digest(&expected_rgba))
            );
            assert_eq!(
                prepared.png().content_sha256(),
                <[u8; 32]>::from(Sha256::digest(prepared.png().encoded_bytes()))
            );
            assert_eq!(
                prepared.png().color_space(),
                graphics_core::resources::ImageColorSpace::Srgb
            );
            assert_eq!(
                prepared.png().alpha(),
                graphics_core::resources::AlphaAssociation::Straight
            );
            assert!(Arc::ptr_eq(
                &prepared.source_output().output_lease(),
                &edited.output().output_lease()
            ));
            assert_eq!(prepared.source_output().parameters(), parameters);
            assert!(Arc::ptr_eq(
                prepared
                    .source_output()
                    .encoded_png_input()
                    .expect("captured source")
                    .source(),
                &source
            ));
            assert_eq!(
                prepared
                    .source_output()
                    .encoded_png_input()
                    .expect("captured placement")
                    .placement()
                    .to_cols_array()
                    .map(f64::to_bits),
                DAffine2::from_scale_angle_translation(DVec2::new(12., 4.), 0., DVec2::new(7., 5.))
                    .to_cols_array()
                    .map(f64::to_bits)
            );
            assert!(session.read_with(cx, |session, _| session.png_is_current(&prepared)));
            assert_eq!(original.output().output(), &original_output);
            assert_ne!(
                session.read_with(cx, |session, _| session.document().network().clone()),
                original_network
            );
            let service = fixture.assets.lock().expect("unchanged service");
            assert_eq!(service.record(&png.asset_identity), Some(input_before));
            assert_eq!(service.record(&output_identity), Some(output_before));
            assert_eq!(
                service
                    .read_verified(
                        &output_identity,
                        &reader,
                        &CancellationToken::default(),
                        16 * 1024
                    )
                    .expect("same indexed bytes"),
                png.encoded
            );
            assert_eq!(receipt, receipt_before);
        }

        #[gpui::test]
        async fn at_r08_r28_committed_artboard_rejects_receipt_joins_and_unauthorized_reader_without_owner(
            cx: &mut TestAppContext,
        ) {
            let fixture = host_reader_fixture().expect("real service");
            let png = indexed_request_png(&fixture, 4).expect("producer PNG");
            let (receipt, _, _, reader) = published(
                &fixture,
                &png,
                "62000000-0000-0000-0000-000000000001",
                "artboard-admission",
            );
            let receipt_before = receipt.clone();
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader.clone()),
                        cx,
                    )
                })
                .expect("separately sealed reader");
            for mismatch in 0..6 {
                let mut request = committed_artboard_request(&png, &receipt);
                let expected = match mismatch {
                    0 => {
                        request.asset_identity.namespace = AssetNamespace::Input;
                        GraphiteAssetBindingError::InvalidOutputReceipt
                    }
                    1 => {
                        request.asset_identity.namespace = AssetNamespace::Temporary;
                        GraphiteAssetBindingError::InvalidOutputReceipt
                    }
                    2 => {
                        request.identity.asset_id = png.image_identity.asset_id.clone();
                        GraphiteAssetBindingError::InvalidOutputReceipt
                    }
                    3 => {
                        request.identity.content_sha256[0] ^= 1;
                        GraphiteAssetBindingError::InvalidOutputReceipt
                    }
                    4 => {
                        request.asset_identity.profile_id = "foreign-artboard-profile".into();
                        GraphiteAssetBindingError::ProfileMismatch
                    }
                    _ => {
                        request.asset_identity = AssetIdentity::new(
                            fixture.input_authorization.profile_id(),
                            AssetNamespace::Output,
                            "uncommitted.png",
                        )
                        .expect("different genuine reference");
                        GraphiteAssetBindingError::InvalidOutputReceipt
                    }
                };
                assert!(
                    matches!(binding.update(cx, |binding, cx| binding.open_committed_artboard(request, &receipt, cx)), Err(error) if error == expected)
                );
                binding.read_with(cx, |binding, _| {
                    assert!(binding.artboard_acquisitions.is_empty());
                    assert!(binding.artboard_sessions.is_empty());
                });
            }
            assert!(matches!(
                binding.update(cx, |binding, cx| binding
                    .open_artboard(committed_artboard_request(&png, &receipt), cx)),
                Err(GraphiteAssetBindingError::InputNamespaceRequired)
            ));
            let missing = cx
                .update(|cx| {
                    GraphiteAssetBinding::new(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("Input-only binding");
            assert!(matches!(
                missing.update(cx, |binding, cx| binding.open_committed_artboard(
                    committed_artboard_request(&png, &receipt),
                    &receipt,
                    cx
                )),
                Err(GraphiteAssetBindingError::MissingOutputAuthorization)
            ));
            missing.read_with(cx, |binding, _| {
                assert!(binding.artboard_acquisitions.is_empty())
            });
            let denied = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(fixture.input_authorization.clone()),
                        cx,
                    )
                })
                .expect("same profile does not imply Output permission");
            let denied = denied
                .update(cx, |binding, cx| {
                    binding.open_committed_artboard(
                        committed_artboard_request(&png, &receipt),
                        &receipt,
                        cx,
                    )
                })
                .expect("real service checks grant")
                .await;
            assert!(
                matches!(denied, Err(GraphiteArtboardOpenError::Read(error)) if matches!(error.as_ref(), crate::resources::GraphiteAssetReadError::Asset(sim_runtime::assets::AssetError::PermissionDenied { namespace: AssetNamespace::Output, .. })))
            );
            let capabilities = CapabilitySet::new([Capability::Asset {
                namespace: AssetNamespace::Output.locator_type().to_owned(),
                action: AssetOperation::Read,
            }]);
            let foreign = sim_runtime::PermissionGrant::new(
                "foreign-artboard-profile",
                "reader",
                capabilities.clone(),
                "component fixture",
            )
            .expect("real foreign grant");
            let foreign = sim_runtime::PermissionPolicy::new("foreign-artboard-profile", [foreign])
                .expect("real policy")
                .authorize("reader", &capabilities)
                .expect("sealed foreign reader");
            assert!(matches!(
                cx.update(|cx| GraphiteAssetBinding::new_with_output_reader(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    Some(foreign),
                    cx
                )),
                Err(GraphiteAssetBindingError::ProfileMismatch)
            ));
            let mut malformed = IndexedRequestPng {
                asset_identity: png.asset_identity.clone(),
                encoded: png.encoded.clone(),
                image_identity: png.image_identity.clone(),
                descriptor: png.descriptor,
            };
            malformed.encoded[0] ^= 1;
            let (malformed_receipt, _, _, _) = published(
                &fixture,
                &malformed,
                "62000000-0000-0000-0000-000000000002",
                "malformed-artboard",
            );
            let mut malformed_request = committed_artboard_request(&malformed, &malformed_receipt);
            malformed_request.identity.content_sha256 = Sha256::digest(&malformed.encoded).into();
            let rejected = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_artboard(malformed_request, &malformed_receipt, cx)
                })
                .expect("actual receipt does not imply valid PNG")
                .await;
            assert!(
                matches!(rejected, Err(GraphiteArtboardOpenError::Read(error)) if matches!(error.as_ref(), crate::resources::GraphiteAssetReadError::Admission(_)))
            );
            binding.read_with(cx, |binding, _| {
                assert!(binding.artboard_acquisitions.is_empty());
                assert!(binding.artboard_sessions.is_empty());
            });
            assert_eq!(receipt, receipt_before);
        }

        #[gpui::test(iterations = 10)]
        async fn at_r08_r28_committed_artboard_invalidation_rejects_buffered_and_png_completion_preserving_edits(
            cx: &mut TestAppContext,
        ) {
            use graphite_glam::{DAffine2, DVec2};
            let fixture = host_reader_fixture().expect("actual service");
            let png = indexed_request_png(&fixture, 4).expect("producer");
            let (receipt, _, _, reader) = published(
                &fixture,
                &png,
                "63000000-0000-0000-0000-000000000001",
                "revoked-artboard",
            );
            let binding = cx
                .update(|cx| {
                    GraphiteAssetBinding::new_with_output_reader(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        Some(reader),
                        cx,
                    )
                })
                .expect("Output read binding");
            let session = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_artboard(
                        committed_artboard_request(&png, &receipt),
                        &receipt,
                        cx,
                    )
                })
                .expect("admission")
                .await
                .expect("real source");
            let source = session.read_with(cx, |session, _| {
                session
                    .document()
                    .encoded_png_input()
                    .expect("retained seal")
                    .source()
                    .clone()
            });
            let mut parameters = session.read_with(cx, |session, _| session.parameters());
            parameters.dimensions = DVec2::new(27., 15.);
            let placement = DAffine2::from_translation(DVec2::new(6., 4.));
            session.update(cx, |session, cx| {
                assert!(
                    session
                        .apply(session.revision(), parameters, session.configuration(), cx)
                        .expect("edit")
                );
                assert!(
                    session
                        .apply_image_placement(session.revision(), placement, cx)
                        .expect("placement")
                );
            });
            let preview = session
                .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
                .expect("request")
                .await
                .expect("real frame");
            let historical_output = preview.output().output().clone();
            let buffered = binding
                .update(cx, |binding, cx| {
                    binding.open_committed_artboard(
                        committed_artboard_request(&png, &receipt),
                        &receipt,
                        cx,
                    )
                })
                .expect("second acquisition");
            cx.run_until_parked();
            let buffered_session = binding.read_with(cx, |binding, _| {
                assert!(binding.artboard_acquisitions.is_empty());
                assert_eq!(binding.artboard_sessions.len(), 2);
                binding
                    .artboard_sessions
                    .last()
                    .expect("buffered owner")
                    .session
                    .upgrade()
                    .expect("receipt retains owner")
            });
            let pending_png = session
                .update(cx, |session, cx| {
                    session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
                })
                .expect("pending PNG");
            binding
                .update(cx, |binding, cx| binding.invalidate(cx))
                .expect("terminal invalidation");
            assert!(matches!(
                buffered.await,
                Err(GraphiteArtboardOpenError::Cancelled)
            ));
            assert!(matches!(
                pending_png.await,
                Err(crate::session::ArtboardSessionError::ResourceRevoked)
            ));
            for owner in [&session, &buffered_session] {
                assert!(owner.read_with(cx, |session, _| session.resource_is_revoked()));
                assert!(matches!(
                    owner.update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx)),
                    Err(crate::session::ArtboardSessionError::ResourceRevoked)
                ));
            }
            session.read_with(cx, |session, _| {
                assert_eq!(session.parameters(), parameters);
                assert_eq!(
                    session
                        .image_placement()
                        .expect("retained placement")
                        .to_cols_array()
                        .map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
                assert!(Arc::ptr_eq(
                    session
                        .document()
                        .encoded_png_input()
                        .expect("retained input")
                        .source(),
                    &source
                ));
                assert!(session.current_output().is_none());
                assert!(!session.preview_is_current(&preview));
            });
            parameters.clip = false;
            session.update(cx, |session, cx| {
                assert!(
                    session
                        .apply(session.revision(), parameters, session.configuration(), cx)
                        .expect("authoring survives revoke")
                );
                assert!(session.resource_is_revoked());
                assert!(session.current_output().is_none());
                assert!(matches!(
                    session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx),
                    Err(crate::session::ArtboardSessionError::ResourceRevoked)
                ));
            });
            assert_eq!(preview.output().output(), &historical_output);
            assert!(preview.image().as_bytes(0).is_some());
        }
    }

    #[cfg(test)]
    mod workspace_tests {
        use super::*;
        use crate::{
            GraphiteMixItem, GraphiteMixSession, ImportedImageItem, InvertImage, MakeImageOpaque,
            ShowOriginalImage,
            resources::tests::{host_reader_fixture, indexed_request_png},
        };
        use gpui::{Action as _, Focusable as _, TestAppContext};
        use graphics_core::resources::{
            CpuBlendMode, CpuImageEdit, CpuMixBudget, CpuMixParameters,
        };
        use project::{FakeFs, Project};
        use std::path::Path;

        fn initialize(cx: &mut TestAppContext) {
            cx.update(|cx| {
                assets::Assets.load_test_fonts(cx);
                let store = settings::SettingsStore::test(cx);
                cx.set_global(store);
                theme_settings::init(theme::LoadThemes::JustBase, cx);
                editor::init(cx);
                cx.set_global(db::AppDatabase::test_new());
                let app_state = workspace::AppState::test(cx);
                workspace::AppState::set_global(app_state, cx);
            });
        }

        #[gpui::test]
        async fn at_graphite_bound_asset_workspace_opens_original_and_edits_without_host_writes(
            cx: &mut TestAppContext,
        ) {
            initialize(cx);
            let fixture = host_reader_fixture().expect("real indexed service");
            let png = indexed_request_png(&fixture, 4).expect("actual PNG producer");
            let before = fixture
                .assets
                .lock()
                .expect("service")
                .record(&png.asset_identity)
                .expect("indexed record");
            assert_eq!(
                before.metadata.workflow.as_deref(),
                Some("{\"fixture\":\"request-resolution\"}")
            );
            let binding = cx
                .update(|cx| {
                    crate::register_asset_binding(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("host-supplied genuine Input grant");
            let fs = FakeFs::new(cx.executor());
            fs.insert_tree(Path::new("/project"), serde_json::json!({}))
                .await;
            let project = Project::test(fs, [Path::new("/project")], cx).await;
            let (host, window) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project, window, cx)
            });
            let workspace = host.read_with(window, |host, _| host.workspace().clone());
            let item = workspace
                .update_in(window, |workspace, window, cx| {
                    crate::open_asset_image(request(&png), 4096, workspace, window, cx)
                })
                .expect("the public host consumer opens the real Item");
            window.run_until_parked();
            assert_eq!(
                workspace.read_with(window, |workspace, cx| workspace
                    .active_item_as::<ImportedImageItem>(cx)
                    .expect("active Item")
                    .entity_id()),
                item.entity_id()
            );
            assert!(
                window.update(|window, cx| item
                    .read(cx)
                    .focus_handle(cx)
                    .contains_focused(window, cx))
            );
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            let session = item.read_with(window, |item, _| item.session().clone());
            let source = session.read_with(window, |session, _| {
                assert!(session.resource_state().is_fresh());
                assert_eq!(session.resource_state().revision(), 1);
                session
                    .resource_state()
                    .committed()
                    .expect("real decode")
                    .clone()
            });
            for (action, operation) in [
                (InvertImage.boxed_clone(), CpuImageEdit::Invert),
                (MakeImageOpaque.boxed_clone(), CpuImageEdit::MakeOpaque),
            ] {
                window.update(|window, cx| window.dispatch_action(action, cx));
                window.run_until_parked();
                session.read_with(window, |session, _| {
                    assert_eq!(session.operation(), Some(operation));
                    assert!(session.result_is_fresh());
                    let expected =
                        graphics_core::resources::evaluate_cpu_image_edit(&source, operation, 4096)
                            .expect("original node oracle");
                    let result = session.result().expect("edited result");
                    assert_eq!(result.image(), expected.image());
                    assert_eq!(result.input_identity(), &png.image_identity);
                    assert!(Arc::ptr_eq(
                        session.resource_state().committed().expect("source"),
                        &source
                    ));
                });
                assert!(window.debug_bounds("GRAPHITE-IMPORTED-RESULT").is_some());
                assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            }
            window.dispatch_action(ShowOriginalImage);
            window.run_until_parked();
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
            session.read_with(window, |session, _| {
                assert_eq!(session.operation(), None);
                assert!(session.result().is_none());
                assert_eq!(session.resource_state().revision(), 1);
            });
            let service = fixture.assets.lock().expect("service");
            assert_eq!(
                service.record(&png.asset_identity).expect("same record"),
                before
            );
            assert_eq!(
                service
                    .read_verified(
                        &png.asset_identity,
                        &fixture.input_authorization,
                        &sim_tensor::CancellationToken::default(),
                        16 * 1024
                    )
                    .expect("original host bytes"),
                png.encoded
            );
            assert!(binding.read_with(window, |binding, _| binding.is_active()));
        }

        #[gpui::test]
        async fn at_graphite_bound_asset_workspace_missing_and_foreign_binding_add_no_item(
            cx: &mut TestAppContext,
        ) {
            initialize(cx);
            let fixture = host_reader_fixture().expect("service");
            let png = indexed_request_png(&fixture, 3).expect("real RGB producer");
            let fs = FakeFs::new(cx.executor());
            fs.insert_tree(Path::new("/project"), serde_json::json!({}))
                .await;
            let project = Project::test(fs, [Path::new("/project")], cx).await;
            let (host, window) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project, window, cx)
            });
            let workspace = host.read_with(window, |host, _| host.workspace().clone());
            assert!(window.read(|cx| crate::asset_binding(cx).is_none()));
            assert!(matches!(
                workspace.update_in(window, |workspace, window, cx| {
                    crate::open_asset_image(request(&png), 4096, workspace, window, cx)
                }),
                Err(GraphiteAssetBindingError::Inactive)
            ));
            assert_eq!(
                workspace.read_with(window, |workspace, cx| workspace
                    .active_pane()
                    .read(cx)
                    .items_len()),
                0
            );
            window
                .update(|_, cx| {
                    crate::register_asset_binding(
                        fixture.assets.clone(),
                        fixture.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("genuine binding");
            let mut foreign = request(&png);
            foreign.asset_identity.profile_id = "foreign-profile".into();
            assert!(matches!(
                workspace.update_in(window, |workspace, window, cx| {
                    crate::open_asset_image(foreign, 4096, workspace, window, cx)
                }),
                Err(GraphiteAssetBindingError::ProfileMismatch)
            ));
            window.run_until_parked();
            assert!(
                workspace.read_with(window, |workspace, cx| workspace.active_item(cx).is_none())
            );
            assert_eq!(
                workspace.read_with(window, |workspace, cx| workspace
                    .active_pane()
                    .read(cx)
                    .items_len()),
                0
            );
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        }

        #[gpui::test]
        async fn at_graphite_bound_asset_workspace_replacement_revokes_existing_item_and_mix(
            cx: &mut TestAppContext,
        ) {
            initialize(cx);
            let first = host_reader_fixture().expect("first real service");
            let png = indexed_request_png(&first, 4).expect("first actual PNG");
            let old_binding = cx
                .update(|cx| {
                    crate::register_asset_binding(
                        first.assets.clone(),
                        first.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("first binding");
            let fs = FakeFs::new(cx.executor());
            fs.insert_tree(Path::new("/project"), serde_json::json!({}))
                .await;
            let project = Project::test(fs, [Path::new("/project")], cx).await;
            let (host, window) = cx.add_window_view(|window, cx| {
                workspace::MultiWorkspace::test_new(project, window, cx)
            });
            let workspace = host.read_with(window, |host, _| host.workspace().clone());
            let item = workspace
                .update_in(window, |workspace, window, cx| {
                    crate::open_asset_image(request(&png), 4096, workspace, window, cx)
                })
                .expect("first Item");
            window.run_until_parked();
            let source = item.read_with(window, |item, _| item.session().clone());
            window.dispatch_action(InvertImage);
            window.run_until_parked();
            assert!(source.read_with(window, |source, _| source.result_is_fresh()));
            let mix = window.new(|cx| {
                GraphiteMixSession::new(
                    [source.clone(), source.clone()],
                    CpuMixParameters {
                        blend_mode: CpuBlendMode::Normal,
                        opacity: 100.,
                    },
                    CpuMixBudget {
                        max_parent_bytes: 4096,
                        max_result_bytes: 4096,
                    },
                    cx,
                )
            });
            window.run_until_parked();
            let mix_item = workspace.update_in(window, |workspace, window, cx| {
                let item =
                    cx.new(|cx| GraphiteMixItem::from_session(mix.clone(), 4096, window, cx));
                workspace.add_item_to_active_pane(Box::new(item.clone()), None, true, window, cx);
                item
            });
            window.run_until_parked();
            assert!(window.debug_bounds("GRAPHITE-MIX-RESULT").is_some());
            assert!(mix_item.read_with(window, |item, cx| {
                item.presentation().read(cx).image().is_some()
            }));
            let replacement = host_reader_fixture().expect("new same-profile host service");
            let next_png = indexed_request_png(&replacement, 3).expect("new indexed PNG");
            assert_eq!(
                first.input_authorization.profile_id(),
                replacement.input_authorization.profile_id()
            );
            assert!(!Arc::ptr_eq(&first.assets, &replacement.assets));
            let new_binding = window
                .update(|_, cx| {
                    crate::register_asset_binding(
                        replacement.assets.clone(),
                        replacement.input_authorization.clone(),
                        cx,
                    )
                })
                .expect("replacement invalidates old lease first");
            assert_ne!(old_binding.entity_id(), new_binding.entity_id());
            window.run_until_parked();
            assert!(!old_binding.read_with(window, |binding, _| binding.is_active()));
            source.read_with(window, |source, _| {
                assert!(source.resource_state().is_revoked());
                assert!(!source.resource_state().is_fresh());
                assert!(source.result().is_none());
            });
            mix.read_with(window, |mix, cx| {
                assert!(mix.is_revoked(cx));
                assert!(!mix.result_is_fresh(cx));
                assert!(mix.result(cx).is_none());
            });
            assert!(window.debug_bounds("GRAPHITE-MIX-REVOKED").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(mix_item.read_with(window, |item, cx| {
                item.presentation().read(cx).image().is_none()
            }));
            workspace.update_in(window, |workspace, window, cx| {
                workspace
                    .active_pane()
                    .update(cx, |pane, cx| pane.activate_item(0, true, true, window, cx))
            });
            window.run_until_parked();
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-REVOKED").is_some());
            assert!(item.read_with(window, |item, cx| {
                item.presentation().read(cx).image().is_none()
            }));
            assert!(matches!(
                old_binding.update(window, |binding, cx| binding.open_image(request(&png), cx)),
                Err(GraphiteAssetBindingError::Inactive)
            ));
            let fresh_item = workspace
                .update_in(window, |workspace, window, cx| {
                    crate::open_asset_image(request(&next_png), 4096, workspace, window, cx)
                })
                .expect("only explicit new binding request recovers");
            window.run_until_parked();
            assert!(window.debug_bounds("GRAPHITE-IMPORTED-ORIGINAL").is_some());
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            fresh_item.read_with(window, |item, cx| {
                let source = item.session().read(cx);
                assert!(source.resource_state().is_fresh());
                assert_eq!(
                    source
                        .resource_state()
                        .committed()
                        .expect("new image")
                        .identity(),
                    &next_png.image_identity
                );
            });
            assert!(source.read_with(window, |source, _| source.resource_state().is_revoked()));
            assert!(mix.read_with(window, |mix, cx| mix.result(cx).is_none()));
        }
    }
}

#[cfg(test)]
pub(crate) mod artboard_binding_controls {
    use super::*;
    use crate::resources::tests::{IndexedRequestPng, host_reader_fixture, indexed_request_png};
    use gpui::TestAppContext;
    use graphite_glam::{DAffine2, DVec2, UVec2};
    use graphite_std::{
        Color,
        application_io::{ExportFormat, RenderConfig, TimingInformation},
        transform::Footprint,
        vector_types::vector::style::RenderMode,
    };

    pub(crate) fn request(png: &IndexedRequestPng) -> GraphiteAssetArtboardRequest {
        GraphiteAssetArtboardRequest {
            asset_identity: png.asset_identity.clone(),
            identity: png.image_identity.clone(),
            placement: DAffine2::IDENTITY,
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
            document_id_encoding: 42,
            node_id: graphite_graph_craft::document::NodeId(7142),
            parameters: CpuSvgArtboardParameters {
                location: DVec2::ZERO,
                dimensions: DVec2::new(31., 17.),
                background: Color::WHITE,
                clip: true,
            },
            configuration: RenderConfig {
                viewport: Footprint {
                    transform: DAffine2::IDENTITY,
                    resolution: UVec2::new(31, 17),
                    quality: Footprint::DEFAULT.quality,
                },
                scale: 1.,
                export_format: ExportFormat::Svg,
                time: TimingInformation {
                    time: 0.,
                    animation_time: std::time::Duration::ZERO,
                },
                pointer: DVec2::ZERO,
                render_mode: RenderMode::Normal,
                for_export: true,
                for_eyedropper: false,
            },
            evaluation_budget: CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
        }
    }

    #[gpui::test(iterations = 10)]
    async fn at_r03_artboard_binding_admits_independent_real_inputs_and_revokes_without_losing_edits(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical asset service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("real Input grant")
        });
        let first = binding
            .update(cx, |binding, cx| binding.open_artboard(request(&png), cx))
            .expect("first request");
        let mut second_request = request(&png);
        second_request.document_id_encoding = 43;
        let second = binding
            .update(cx, |binding, cx| binding.open_artboard(second_request, cx))
            .expect("independent request");
        let first = first.await.expect("first source admitted");
        let second = second.await.expect("second source admitted");
        first.read_with(cx, |session, _| {
            assert_eq!(session.document().document_id_encoding(), 42);
            assert_eq!(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source")
                    .source()
                    .identity(),
                &png.image_identity
            );
            assert_eq!(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source")
                    .source()
                    .width(),
                png.descriptor.width
            );
            assert_eq!(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source")
                    .source()
                    .height(),
                png.descriptor.height
            );
            assert!(!session.resource_is_revoked());
        });
        second.read_with(cx, |session, _| {
            assert_eq!(session.document().document_id_encoding(), 43)
        });
        let source = first.read_with(cx, |session, _| {
            session
                .document()
                .encoded_png_input()
                .expect("source")
                .source()
                .clone()
        });
        let mut parameters = first.read_with(cx, |session, _| session.parameters());
        parameters.dimensions = DVec2::new(40., 20.);
        first.update(cx, |session, cx| {
            assert!(
                session
                    .apply(session.revision(), parameters, session.configuration(), cx)
                    .expect("editable source")
            )
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("terminal invalidation");
        for session in [&first, &second] {
            session.read_with(cx, |session, _| {
                assert!(session.resource_is_revoked());
                assert!(session.current_output().is_none());
                assert!(session.preview().is_none());
                assert!(matches!(
                    session.diagnostic(),
                    Some(crate::session::ArtboardSessionError::ResourceRevoked)
                ));
            });
            assert!(matches!(
                session.update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx)),
                Err(crate::session::ArtboardSessionError::ResourceRevoked)
            ));
        }
        first.update(cx, |session, cx| {
            parameters.clip = false;
            assert!(
                session
                    .apply(session.revision(), parameters, session.configuration(), cx)
                    .expect("preserve authoring after revoke")
            );
            assert!(session.resource_is_revoked());
            assert!(matches!(
                session.diagnostic(),
                Some(crate::session::ArtboardSessionError::ResourceRevoked)
            ));
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("retained source")
                    .source(),
                &source
            ));
        });
        assert!(matches!(
            binding.update(cx, |binding, cx| binding.open_artboard(request(&png), cx)),
            Err(GraphiteAssetBindingError::Inactive)
        ));
    }

    #[gpui::test(iterations = 10)]
    fn at_r08_unpolled_artboard_receipt_abandonment_and_binding_invalidation_cancel_acquisition(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("matching grant")
        });
        let receipt = binding
            .update(cx, |binding, cx| binding.open_artboard(request(&png), cx))
            .expect("pending request");
        let cancellation = binding.read_with(cx, |binding, _| {
            binding
                .artboard_acquisitions
                .first()
                .expect("synchronous token registration")
                .cancellation
                .clone()
        });
        drop(receipt);
        cx.run_until_parked();
        assert!(cancellation.is_cancelled());
        binding.read_with(cx, |binding, _| {
            assert!(binding.artboard_acquisitions.is_empty());
            assert!(binding.artboard_sessions.is_empty());
        });
        let receipt = binding
            .update(cx, |binding, cx| binding.open_artboard(request(&png), cx))
            .expect("second pending request");
        let cancellation = binding.read_with(cx, |binding, _| {
            binding
                .artboard_acquisitions
                .first()
                .expect("registered")
                .cancellation
                .clone()
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("invalidate before any scheduling");
        assert!(cancellation.is_cancelled());
        binding.read_with(cx, |binding, _| {
            assert!(
                binding.artboard_acquisitions.is_empty() && binding.artboard_sessions.is_empty()
            )
        });
        drop(receipt);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn at_r08_failed_real_artboard_read_never_installs_a_session(cx: &mut TestAppContext) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("matching grant")
        });
        let mut wrong = request(&png);
        wrong.identity.content_sha256[0] ^= 1;
        let outcome = binding
            .update(cx, |binding, cx| binding.open_artboard(wrong, cx))
            .expect("valid binding")
            .await;
        assert!(matches!(outcome, Err(GraphiteArtboardOpenError::Read(_))));
        binding.read_with(cx, |binding, _| {
            assert!(
                binding.artboard_acquisitions.is_empty() && binding.artboard_sessions.is_empty()
            )
        });
        let mut wrong = request(&png);
        wrong.configuration.scale = f64::NAN;
        let outcome = binding
            .update(cx, |binding, cx| binding.open_artboard(wrong, cx))
            .expect("valid binding")
            .await;
        assert!(matches!(
            outcome,
            Err(GraphiteArtboardOpenError::Document(
                CpuSvgDocumentError::InvalidConfiguration
            ))
        ));
        binding.read_with(cx, |binding, _| {
            assert!(
                binding.artboard_acquisitions.is_empty() && binding.artboard_sessions.is_empty()
            )
        });
    }

    #[gpui::test(iterations = 10)]
    async fn at_r08_buffered_artboard_completion_observes_later_binding_revocation(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("matching grant")
        });
        let receipt = binding
            .update(cx, |binding, cx| binding.open_artboard(request(&png), cx))
            .expect("admitted request");
        cx.run_until_parked();
        binding.read_with(cx, |binding, _| {
            assert!(binding.artboard_acquisitions.is_empty());
            assert_eq!(binding.artboard_sessions.len(), 1);
            assert!(
                !binding
                    .artboard_sessions
                    .first()
                    .expect("tracked before delivery")
                    .session
                    .upgrade()
                    .expect("receipt retains session")
                    .read_with(cx, |session, _| session.resource_is_revoked())
            );
        });
        let session = binding.read_with(cx, |binding, _| {
            binding
                .artboard_sessions
                .first()
                .expect("tracked session")
                .session
                .upgrade()
                .expect("buffered receipt retains source")
        });
        binding
            .update(cx, |binding, cx| binding.invalidate(cx))
            .expect("revoke after completed acquisition");
        assert!(matches!(
            receipt.await,
            Err(GraphiteArtboardOpenError::Cancelled)
        ));
        assert!(session.read_with(cx, |session, _| session.resource_is_revoked()));
        assert!(matches!(
            session.update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx)),
            Err(crate::session::ArtboardSessionError::ResourceRevoked)
        ));
        assert!(session.read_with(cx, |session, _| {
            session.document().encoded_png_input().is_some()
        }));
    }

    #[gpui::test(iterations = 10)]
    async fn at_r08_binding_release_cancels_unpolled_artboard_acquisition(cx: &mut TestAppContext) {
        let fixture = host_reader_fixture().expect("canonical service");
        let png = indexed_request_png(&fixture, 4).expect("indexed PNG");
        let binding = cx.update(|cx| {
            GraphiteAssetBinding::new(
                fixture.assets.clone(),
                fixture.input_authorization.clone(),
                cx,
            )
            .expect("matching grant")
        });
        let receipt = binding
            .update(cx, |binding, cx| binding.open_artboard(request(&png), cx))
            .expect("pending request");
        let cancellation = binding.read_with(cx, |binding, _| {
            binding
                .artboard_acquisitions
                .first()
                .expect("registered")
                .cancellation
                .clone()
        });
        let owner = binding.downgrade();
        drop(binding);
        cx.run_until_parked();
        assert!(owner.upgrade().is_none());
        assert!(cancellation.is_cancelled());
        assert!(matches!(
            receipt.await,
            Err(GraphiteArtboardOpenError::Cancelled)
        ));
    }
}

#[derive(Clone)]
pub(crate) struct GraphiteProjectContext {
    binding: WeakEntity<GraphiteAssetBinding>,
    pub(crate) assets: SharedAssetService,
    pub(crate) input_reader: AuthorizedCapabilities,
    pub(crate) output_reader: AuthorizedCapabilities,
    pub(crate) committer: sim_runtime::SharedOutputCommitter,
    pub(crate) publisher: AuthorizedCapabilities,
}

impl GraphiteProjectContext {
    pub(crate) fn track_operation(
        &self,
        cancellation: &Arc<CancellationToken>,
        cx: &mut App,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(self.current(cx), "The project asset profile changed.");
        self.binding.update(cx, |binding, _| {
            binding
                .project_operations
                .retain(|token| token.strong_count() > 0);
            binding
                .project_operations
                .push(Arc::downgrade(cancellation));
        })?;
        Ok(())
    }

    pub(crate) fn current(&self, cx: &App) -> bool {
        self.binding.upgrade().is_some_and(|binding| {
            crate::asset_binding(cx)
                .is_some_and(|current| current.entity_id() == binding.entity_id())
                && binding.read(cx).is_current_with_output_reader(
                    &self.assets,
                    &self.input_reader,
                    Some(&self.output_reader),
                )
                && binding
                    .read(cx)
                    .publisher
                    .as_ref()
                    .is_some_and(|(committer, authorization)| {
                        Arc::ptr_eq(committer, &self.committer) && *authorization == self.publisher
                    })
        })
    }

    pub(crate) fn publish(
        &self,
        proposal: sim_runtime::OutputProposal,
        cancellation: &CancellationToken,
    ) -> anyhow::Result<OutputCommitReceipt> {
        let mut assets = self.assets.try_lock().map_err(|_| {
            anyhow::anyhow!("The project asset service is busy. Retry after the current job.")
        })?;
        let mut committer = self.committer.try_lock().map_err(|_| {
            anyhow::anyhow!("The project publisher is busy. Retry after the current job.")
        })?;
        let receipts = committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::Local::now().fixed_offset(),
            &mut assets,
            &self.publisher,
            cancellation,
        )?;
        receipts
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("The publisher returned no committed output."))
    }
}

#[derive(Clone)]
pub(crate) struct GraphiteLinkedDocument {
    pub(crate) project: GraphiteProjectContext,
    pub(crate) container: sim_runtime::SimDocument,
    pub(crate) engine_index: usize,
    pub(crate) input_version: sim_runtime::ProjectAssetVersion,
    pub(crate) resource_id: graphite_graph_craft::application_io::resource::ResourceId,
    pub(crate) saved_source: Option<sim_runtime::ProjectEditableSourceReference>,
    pub(crate) saved_revision: Option<u64>,
}

impl GraphiteAssetBinding {
    pub(crate) fn set_publisher(
        &mut self,
        committer: sim_runtime::SharedOutputCommitter,
        authorization: AuthorizedCapabilities,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.active && authorization.profile_id() == self.authorization.profile_id(),
            "The project publisher profile does not match the active reader."
        );
        let assets = self
            .assets
            .lock()
            .map_err(|_| anyhow::anyhow!("The project asset service is unavailable."))?;
        let writer = committer
            .lock()
            .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
        anyhow::ensure!(
            assets.roots() == writer.roots(),
            "The project publisher roots do not match the active reader."
        );
        if self
            .publisher
            .as_ref()
            .is_some_and(|(existing, capabilities)| {
                !Arc::ptr_eq(existing, &committer) || *capabilities != authorization
            })
        {
            for cancellation in self
                .project_operations
                .drain(..)
                .filter_map(|token| token.upgrade())
            {
                cancellation.cancel();
            }
        }
        self.publisher = Some((committer.clone(), authorization));
        Ok(())
    }

    pub(crate) fn project_context(
        &self,
        cx: &Context<Self>,
    ) -> anyhow::Result<GraphiteProjectContext> {
        anyhow::ensure!(self.active, "The project asset reader is inactive.");
        let (committer, publisher) = self.publisher.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "This project has no authorized publisher. Activate a native project profile."
            )
        })?;
        Ok(GraphiteProjectContext {
            binding: cx.entity().downgrade(),
            assets: self.assets.clone(),
            input_reader: self.authorization.clone(),
            output_reader: self
                .output_authorization
                .clone()
                .ok_or_else(|| anyhow::anyhow!("This project has no Output read permission."))?,
            committer,
            publisher,
        })
    }

    pub(crate) fn bind_linked_session(
        &mut self,
        document: CpuSvgArtboardDocument,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<Entity<GraphiteArtboardSession>> {
        anyhow::ensure!(self.active, "The project asset reader is inactive.");
        let session = cx.new(|_| GraphiteArtboardSession::new(document));
        self.artboard_sessions
            .retain(|bound| bound.session.upgrade().is_some());
        self.artboard_sessions.push(BoundArtboardSession {
            session: session.downgrade(),
            cancellation: Arc::new(CancellationToken::default()),
        });
        Ok(session)
    }
}

#[derive(Clone)]
pub struct GraphiteSimDocumentChoice {
    binding: WeakEntity<GraphiteAssetBinding>,
    record: sim_runtime::assets::AssetRecord,
}

impl GraphiteSimDocumentChoice {
    pub fn label(&self) -> String {
        self.record
            .identity
            .relative_path
            .to_string_lossy()
            .into_owned()
    }
    pub fn byte_size(&self) -> u64 {
        self.record.byte_size
    }
    pub(crate) fn binding(&self) -> anyhow::Result<Entity<GraphiteAssetBinding>> {
        self.binding
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("The project reader was closed. Reopen the chooser."))
    }
}

pub struct GraphiteSimDocumentPage {
    pub choices: Vec<GraphiteSimDocumentChoice>,
    pub next_offset: Option<usize>,
}

pub(crate) fn query_saved_sim_documents(
    offset: usize,
    text: Option<String>,
    cx: &mut App,
) -> anyhow::Result<Task<anyhow::Result<GraphiteSimDocumentPage>>> {
    let binding = crate::asset_binding(cx).ok_or_else(|| {
        anyhow::anyhow!(
            "The project reader is unavailable. Activate a native project profile and retry."
        )
    })?;
    let project = binding.update(cx, |binding, cx| binding.project_context(cx))?;
    let weak_binding = binding.downgrade();
    let background = cx.background_spawn({
        let project = project.clone();
        async move {
            let assets = project
                .assets
                .lock()
                .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
            Ok::<_, anyhow::Error>(assets.list_authorized(
                &sim_runtime::assets::AssetQuery {
                    namespace: Some(AssetNamespace::Output),
                    text,
                    offset,
                    ..Default::default()
                },
                &project.output_reader,
            )?)
        }
    });
    Ok(cx.spawn(async move |cx| {
        let page = background.await?;
        let current = cx.update(|cx| project.current(cx));
        anyhow::ensure!(current, "The project profile changed. Reopen the chooser.");
        Ok(GraphiteSimDocumentPage {
            choices: page
                .records
                .into_iter()
                .filter(|record| {
                    record
                        .identity
                        .relative_path
                        .extension()
                        .and_then(|value| value.to_str())
                        == Some("sim")
                        && record.availability != sim_runtime::assets::AssetAvailability::Missing
                })
                .map(|record| GraphiteSimDocumentChoice {
                    binding: weak_binding.clone(),
                    record,
                })
                .collect(),
            next_offset: page.next_offset,
        })
    }))
}

pub(crate) fn acquire_linked_png(
    choice: GraphitePngAssetChoice,
    project: GraphiteProjectContext,
    cancellation: Arc<CancellationToken>,
) -> anyhow::Result<(CpuSvgArtboardDocument, GraphiteLinkedDocument)> {
    let request = choice.request()?;
    let version = {
        let mut assets = project
            .assets
            .lock()
            .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
        let mut committer = project
            .committer
            .lock()
            .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
        assets.pin_project_png_version(
            &mut committer,
            &choice.record.identity,
            &choice.record.sha256,
            &project.input_reader,
            &project.output_reader,
            &project.publisher,
            &cancellation,
            request.encoded_budget.max_encoded_bytes as u64,
        )?
    };
    create_linked_png_artboard(request, version, project, &cancellation)
}

pub(crate) fn acquire_published_png(
    choice: GraphitePublishedPngChoice,
    project: GraphiteProjectContext,
    cancellation: Arc<CancellationToken>,
) -> anyhow::Result<(CpuSvgArtboardDocument, GraphiteLinkedDocument)> {
    let request = choice.choice.request()?;
    {
        let assets = project
            .assets
            .lock()
            .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
        let committer = project
            .committer
            .lock()
            .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
        let current = assets
            .project_png_versions(&project.output_reader)?
            .into_iter()
            .find(|version| version.reference == choice.version.reference)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "The selected immutable PNG version is missing. Reopen the chooser."
                )
            })?;
        anyhow::ensure!(
            current == choice.version,
            "The selected project version changed. Reopen the chooser."
        );
        verify_published_version(&committer, &current)?;
        anyhow::ensure!(
            request.identity.content_sha256
                == decode_version_digest(&current.reference.content_sha256)?,
            "The indexed PNG does not match the selected immutable version. Reopen the chooser."
        );
    }
    create_linked_png_artboard(request, choice.version, project, &cancellation)
}

fn create_linked_png_artboard(
    mut request: GraphiteAssetArtboardRequest,
    version: sim_runtime::ProjectAssetVersion,
    project: GraphiteProjectContext,
    cancellation: &CancellationToken,
) -> anyhow::Result<(CpuSvgArtboardDocument, GraphiteLinkedDocument)> {
    request.identity.asset_id = version.reference.asset_id.to_string().into();
    request.identity.version_id = version.reference.version_id.to_string().into();
    request.identity.content_sha256 = decode_version_digest(&version.reference.content_sha256)?;
    let source = {
        let assets = project
            .assets
            .lock()
            .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
        {
            let committer = project
                .committer
                .lock()
                .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
            verify_published_closure(
                &committer,
                &assets.project_png_versions(&project.output_reader)?,
                &version,
                &assets,
                &project.output_reader,
                cancellation,
            )?;
        }
        let source = assets.resolve_project_asset_version(
            &version.reference,
            &project.output_reader,
            cancellation,
            request.encoded_budget.max_encoded_bytes as u64,
        )?;
        anyhow::ensure!(
            source.version() == &version,
            "The selected project version changed. Reopen the chooser."
        );
        source
    };
    let prepared = graphics_core::resources::prepare_encoded_png_source(
        request.identity,
        Arc::from(source.bytes()),
        request.encoded_budget,
    )?;
    let document = CpuSvgArtboardDocument::new_with_encoded_png(
        request.document_id_encoding,
        request.node_id,
        request.parameters,
        request.configuration,
        request.evaluation_budget,
        CpuSvgEncodedPngInput::new(Arc::new(prepared), request.placement)?,
    )?;
    let project_id = uuid::Uuid::parse_str(&version.stored_identity.profile_id)?;
    let container = sim_runtime::SimDocument {
        format: "sim".into(),
        schema_version: sim_runtime::SIM_DOCUMENT_SCHEMA_VERSION,
        project_id,
        documents: vec![sim_runtime::SimEngineDocument::new(
            uuid::Uuid::new_v4(),
            "graphite".into(),
            graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION.into(),
            "sim.graphite.legacy-artboard.v1".into(),
            String::new(),
            vec![version.reference.clone()],
        )],
        build_outputs: vec![],
        provenance: serde_json::json!({"kind":"graphite-png-composition", "original_identity":version.original_identity, "input_version":version.reference, "producer_provenance":version.provenance, "producer_editable_source":version.editable_source}),
    };
    Ok((
        document,
        GraphiteLinkedDocument {
            project,
            container,
            engine_index: 0,
            input_version: version,
            resource_id: graphite_graph_craft::application_io::resource::ResourceId::from(
                request.document_id_encoding,
            ),
            saved_source: None,
            saved_revision: None,
        },
    ))
}

pub(crate) fn acquire_saved_sim(
    choice: GraphiteSimDocumentChoice,
    project: GraphiteProjectContext,
    cancellation: Arc<CancellationToken>,
) -> anyhow::Result<(CpuSvgArtboardDocument, GraphiteLinkedDocument)> {
    let (bytes, version, container, engine_index) = {
        let assets = project
            .assets
            .lock()
            .map_err(|_| anyhow::anyhow!("The project reader is unavailable."))?;
        let bytes = assets.read_verified(
            &choice.record.identity,
            &project.output_reader,
            &cancellation,
            sim_runtime::MAX_SIM_DOCUMENT_BYTES as u64,
        )?;
        anyhow::ensure!(
            encoded_sha256(&bytes) == choice.record.sha256,
            "The saved .sim changed. Reopen the chooser."
        );
        let container = sim_runtime::SimDocument::decode(&bytes)?;
        anyhow::ensure!(
            container.project_id.to_string() == project.output_reader.profile_id(),
            "The .sim document belongs to another project."
        );
        let engines: Vec<_> = container
            .documents
            .iter()
            .enumerate()
            .filter(|(_, document)| document.engine == "graphite")
            .collect();
        anyhow::ensure!(
            engines.len() == 1,
            "This PNG editor requires exactly one Graphite document; other engine documents remain preserved."
        );
        let (engine_index, engine) = engines[0];
        anyhow::ensure!(
            engine.source_revision == graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION
                && engine.serialization_format == "sim.graphite.legacy-artboard.v1"
                && engine.dependencies.len() == 1,
            "This Graphite payload format or source pin is unsupported."
        );
        for document in &container.documents {
            for reference in &document.dependencies {
                assets.resolve_project_asset_version(
                    reference,
                    &project.output_reader,
                    &cancellation,
                    64 * 1024 * 1024,
                )?;
            }
        }
        for reference in &container.build_outputs {
            assets.resolve_project_asset_version(
                reference,
                &project.output_reader,
                &cancellation,
                64 * 1024 * 1024,
            )?;
        }
        let version = assets.resolve_project_asset_version(
            &engine.dependencies[0],
            &project.output_reader,
            &cancellation,
            64 * 1024 * 1024,
        )?;
        {
            let committer = project
                .committer
                .lock()
                .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
            verify_published_closure(
                &committer,
                &assets.project_png_versions(&project.output_reader)?,
                version.version(),
                &assets,
                &project.output_reader,
                &cancellation,
            )?;
        }
        (bytes, version, container, engine_index)
    };
    let mut digest = [0u8; 32];
    for (target, pair) in digest.iter_mut().zip(
        version
            .version()
            .reference
            .content_sha256
            .as_bytes()
            .as_chunks::<2>()
            .0,
    ) {
        *target = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    let prepared = Arc::new(graphics_core::resources::prepare_encoded_png_source(
        ImageResourceIdentity {
            asset_id: version.version().reference.asset_id.to_string().into(),
            version_id: version.version().reference.version_id.to_string().into(),
            content_sha256: digest,
        },
        Arc::from(version.bytes()),
        EncodedPngBudget {
            max_encoded_bytes: 64 * 1024 * 1024,
            max_pixel_bytes: 128 * 1024 * 1024,
            max_metadata_bytes: 1024 * 1024,
        },
    )?);
    let restored = graphics_core::formats::reopen_artboard_engine_payload_with_resource(
        container.documents[engine_index].payload.as_bytes(),
        prepared,
        CpuSvgEvaluationBudget {
            max_viewport_pixels: 16 * 1024 * 1024,
            max_output_bytes: 64 * 1024 * 1024,
        },
        sim_runtime::MAX_SIM_DOCUMENT_BYTES,
    )?;
    let resource_id = restored.resource_id();
    let document = restored.into_document();
    let saved_revision = Some(document.revision());
    let saved_source = Some(sim_runtime::ProjectEditableSourceReference {
        identity: choice.record.identity,
        content_sha256: encoded_sha256(&bytes),
    });
    Ok((
        document,
        GraphiteLinkedDocument {
            project,
            container,
            engine_index,
            input_version: version.version().clone(),
            resource_id,
            saved_source,
            saved_revision,
        },
    ))
}

fn decode_version_digest(encoded: &str) -> anyhow::Result<[u8; 32]> {
    anyhow::ensure!(
        encoded.len() == 64,
        "The project PNG version has an invalid digest."
    );
    let mut digest = [0u8; 32];
    for (target, pair) in digest.iter_mut().zip(encoded.as_bytes().as_chunks::<2>().0) {
        *target = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(digest)
}

fn encoded_sha256(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

pub(crate) fn prepare_linked_input_update(
    linked: &GraphiteLinkedDocument,
    cancellation: &CancellationToken,
) -> anyhow::Result<(
    sim_runtime::ProjectAssetVersion,
    Arc<graphics_core::resources::PreparedEncodedPngSource>,
)> {
    let mut assets = linked
        .project
        .assets
        .lock()
        .map_err(|_| anyhow::anyhow!("The project asset service is unavailable."))?;
    let original = &linked.input_version.original_identity;
    let reader = if original.namespace == AssetNamespace::Input {
        &linked.project.input_reader
    } else {
        &linked.project.output_reader
    };
    let version = if !linked
        .input_version
        .stored_identity
        .relative_path
        .starts_with(".sim/inputs")
    {
        let version = assets.project_png_versions(&linked.project.output_reader)?.into_iter()
            .rev()
            .find(|version| version.reference.asset_id == linked.input_version.reference.asset_id)
            .ok_or_else(|| anyhow::anyhow!("The published PNG version is missing; the pinned version and edits remain preserved."))?;
        let committer = linked
            .project
            .committer
            .lock()
            .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
        verify_published_closure(
            &committer,
            &assets.project_png_versions(&linked.project.output_reader)?,
            &version,
            &assets,
            &linked.project.output_reader,
            cancellation,
        )?;
        version
    } else {
        assets.scan_namespaces(&[original.namespace], reader, cancellation)?;
        let record = assets.record(original).ok_or_else(|| {
            anyhow::anyhow!(
                "The original PNG is missing; the pinned version and edits remain preserved."
            )
        })?;
        let mut committer = linked
            .project
            .committer
            .lock()
            .map_err(|_| anyhow::anyhow!("The project publisher is unavailable."))?;
        assets.pin_project_png_version(
            &mut committer,
            original,
            &record.sha256,
            reader,
            &linked.project.output_reader,
            &linked.project.publisher,
            cancellation,
            64 * 1024 * 1024,
        )?
    };
    let source = assets.resolve_project_asset_version(
        &version.reference,
        &linked.project.output_reader,
        cancellation,
        64 * 1024 * 1024,
    )?;
    anyhow::ensure!(
        source.version() == &version,
        "The published PNG version changed; the pinned version and edits remain preserved."
    );
    let mut digest = [0u8; 32];
    for (target, pair) in digest.iter_mut().zip(
        version
            .reference
            .content_sha256
            .as_bytes()
            .as_chunks::<2>()
            .0,
    ) {
        *target = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    let prepared = graphics_core::resources::prepare_encoded_png_source(
        ImageResourceIdentity {
            asset_id: version.reference.asset_id.to_string().into(),
            version_id: version.reference.version_id.to_string().into(),
            content_sha256: digest,
        },
        Arc::from(source.bytes()),
        EncodedPngBudget {
            max_encoded_bytes: 64 * 1024 * 1024,
            max_pixel_bytes: 128 * 1024 * 1024,
            max_metadata_bytes: 1024 * 1024,
        },
    )?;
    Ok((version, Arc::new(prepared)))
}
