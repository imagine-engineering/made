pub mod image_preview;
pub mod interop;
pub mod resources;
pub mod storage;

mod graph;
mod png_workflow;
mod session;

pub use graph::{
    CancelImageLoad, CancelMixEvaluation, GraphicsWorkspaceItem, GraphiteArtboardItem,
    GraphiteMixItem, ImportedImageItem, InvertImage, MakeImageOpaque, RetryMixEvaluation,
    ShowOriginalImage, bind_keys, open_numeric_document,
};
pub use session::{
    ArtboardSessionError, ArtboardSessionEvent, GraphicsSession, GraphiteArtboardSession,
    GraphiteEncodedPngLoad, GraphiteImageSession, GraphiteMixSession, GraphitePngLoad,
    ImageSessionDiagnostic, ImageSessionEvent, NumericEvaluationCompletion,
    NumericEvaluationRequest, OriginalDocumentCapture, PreparedArtboardPngAsset,
    PreparedArtboardPreview,
};

#[cfg(test)]
mod conformance;

struct AssetBindingRegistration(gpui::Entity<interop::GraphiteAssetBinding>);

impl gpui::Global for AssetBindingRegistration {}

pub fn asset_binding(cx: &gpui::App) -> Option<gpui::Entity<interop::GraphiteAssetBinding>> {
    cx.try_global::<AssetBindingRegistration>()
        .map(|registration| registration.0.clone())
}

pub fn register_asset_binding(
    assets: sim_runtime::assets::SharedAssetService,
    authorization: sim_runtime::AuthorizedCapabilities,
    cx: &mut gpui::App,
) -> Result<gpui::Entity<interop::GraphiteAssetBinding>, interop::GraphiteAssetBindingError> {
    register_asset_binding_with_output_reader(assets, authorization, None, cx)
}

pub fn register_asset_binding_with_output_reader(
    assets: sim_runtime::assets::SharedAssetService,
    authorization: sim_runtime::AuthorizedCapabilities,
    output_authorization: Option<sim_runtime::AuthorizedCapabilities>,
    cx: &mut gpui::App,
) -> Result<gpui::Entity<interop::GraphiteAssetBinding>, interop::GraphiteAssetBindingError> {
    if let Some(binding) = asset_binding(cx) {
        if binding.read(cx).is_current_with_output_reader(
            &assets,
            &authorization,
            output_authorization.as_ref(),
        ) {
            return Ok(binding);
        }
        clear_asset_binding(cx)?;
    }
    let binding = interop::GraphiteAssetBinding::new_with_output_reader(
        assets,
        authorization,
        output_authorization,
        cx,
    )?;
    cx.set_global(AssetBindingRegistration(binding.clone()));
    Ok(binding)
}

pub fn clear_asset_binding(cx: &mut gpui::App) -> Result<(), interop::GraphiteAssetBindingError> {
    if cx.has_global::<AssetBindingRegistration>() {
        let registration = cx.remove_global::<AssetBindingRegistration>();
        registration
            .0
            .update(cx, |binding, cx| binding.invalidate(cx))?;
    }
    Ok(())
}

pub fn open_asset_image(
    request: interop::GraphiteAssetImageRequest,
    max_preview_bytes: usize,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<gpui::Entity<ImportedImageItem>, interop::GraphiteAssetBindingError> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let session = binding.update(cx, |binding, cx| binding.open_image(request, cx))?;
    Ok(add_asset_image_session(
        session,
        max_preview_bytes,
        None,
        workspace,
        window,
        cx,
    ))
}

pub fn open_committed_asset_image(
    request: interop::GraphiteAssetImageRequest,
    receipt: &sim_runtime::OutputCommitReceipt,
    max_preview_bytes: usize,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<gpui::Entity<ImportedImageItem>, interop::GraphiteAssetBindingError> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let session = binding.update(cx, |binding, cx| {
        binding.open_committed_image(request, receipt, cx)
    })?;
    Ok(add_asset_image_session(
        session,
        max_preview_bytes,
        None,
        workspace,
        window,
        cx,
    ))
}

pub fn open_asset_image_with_scene_configuration(
    request: interop::GraphiteAssetImageRequest,
    max_preview_bytes: usize,
    configuration: graphics_core::render::CpuSvgSceneConfiguration,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<gpui::Entity<ImportedImageItem>, interop::GraphiteAssetBindingError> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let session = binding.update(cx, |binding, cx| binding.open_image(request, cx))?;
    Ok(add_asset_image_session(
        session,
        max_preview_bytes,
        Some(configuration),
        workspace,
        window,
        cx,
    ))
}

pub fn open_committed_asset_image_with_scene_configuration(
    request: interop::GraphiteAssetImageRequest,
    receipt: &sim_runtime::OutputCommitReceipt,
    max_preview_bytes: usize,
    configuration: graphics_core::render::CpuSvgSceneConfiguration,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<gpui::Entity<ImportedImageItem>, interop::GraphiteAssetBindingError> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let session = binding.update(cx, |binding, cx| {
        binding.open_committed_image(request, receipt, cx)
    })?;
    Ok(add_asset_image_session(
        session,
        max_preview_bytes,
        Some(configuration),
        workspace,
        window,
        cx,
    ))
}

fn add_asset_image_session(
    session: gpui::Entity<GraphiteImageSession>,
    max_preview_bytes: usize,
    configuration: Option<graphics_core::render::CpuSvgSceneConfiguration>,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> gpui::Entity<ImportedImageItem> {
    use gpui::AppContext as _;

    let item = cx.new(|cx| match configuration {
        Some(configuration) => ImportedImageItem::from_session_with_scene_configuration(
            session,
            max_preview_bytes,
            configuration,
            window,
            cx,
        ),
        None => ImportedImageItem::from_session(session, max_preview_bytes, window, cx),
    });
    workspace.active_pane().update(cx, |pane, cx| {
        pane.add_item(Box::new(item.clone()), true, true, None, window, cx);
    });
    item
}

pub mod render;

pub fn open_artboard_document(
    document: graphics_core::evaluate::CpuSvgArtboardDocument,
    max_preview_bytes: usize,
    workspace: &mut workspace::Workspace,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> gpui::Entity<GraphiteArtboardItem> {
    use gpui::AppContext as _;

    let session = cx.new(|_| GraphiteArtboardSession::new(document));
    let item =
        cx.new(|cx| GraphiteArtboardItem::from_session(session, max_preview_bytes, window, cx));
    workspace.active_pane().update(cx, |pane, cx| {
        pane.add_item(Box::new(item.clone()), true, true, None, window, cx);
    });
    item
}

pub fn open_asset_artboard(
    request: interop::GraphiteAssetArtboardRequest,
    max_preview_bytes: usize,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<
    gpui::Task<Result<gpui::Entity<GraphiteArtboardItem>, interop::GraphiteArtboardOpenError>>,
    interop::GraphiteAssetBindingError,
> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let acquisition = binding.update(cx, |binding, cx| binding.open_artboard(request, cx))?;
    Ok(add_asset_artboard_acquisition(
        acquisition,
        max_preview_bytes,
        window,
        cx,
    ))
}

pub fn open_committed_asset_artboard(
    request: interop::GraphiteAssetArtboardRequest,
    receipt: &sim_runtime::OutputCommitReceipt,
    max_preview_bytes: usize,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> Result<
    gpui::Task<Result<gpui::Entity<GraphiteArtboardItem>, interop::GraphiteArtboardOpenError>>,
    interop::GraphiteAssetBindingError,
> {
    let binding = asset_binding(cx).ok_or(interop::GraphiteAssetBindingError::Inactive)?;
    let acquisition = binding.update(cx, |binding, cx| {
        binding.open_committed_artboard(request, receipt, cx)
    })?;
    Ok(add_asset_artboard_acquisition(
        acquisition,
        max_preview_bytes,
        window,
        cx,
    ))
}

fn add_asset_artboard_acquisition(
    acquisition: interop::GraphiteArtboardAcquisition,
    max_preview_bytes: usize,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> gpui::Task<Result<gpui::Entity<GraphiteArtboardItem>, interop::GraphiteArtboardOpenError>> {
    use gpui::AppContext as _;
    cx.spawn_in(window, async move |workspace, cx| {
        let session = acquisition.await?;
        workspace
            .update_in(cx, |workspace, window, cx| {
                if session.read(cx).resource_is_revoked() {
                    return Err(interop::GraphiteArtboardOpenError::BindingRevoked);
                }
                let item = cx.new(|cx| {
                    GraphiteArtboardItem::from_session(session, max_preview_bytes, window, cx)
                });
                workspace.active_pane().update(cx, |pane, cx| {
                    pane.add_item(Box::new(item.clone()), true, true, None, window, cx);
                });
                Ok(item)
            })
            .map_err(|_| interop::GraphiteArtboardOpenError::Released)?
    })
}

pub fn init(cx: &mut gpui::App) {
    graph::init(cx);
    png_workflow::init(cx);
}

pub fn query_authorized_pngs(
    offset: usize,
    text: Option<String>,
    cx: &mut gpui::App,
) -> anyhow::Result<gpui::Task<anyhow::Result<interop::GraphitePngAssetPage>>> {
    interop::query_authorized_pngs(offset, text, cx)
}

pub fn register_asset_publisher(
    committer: sim_runtime::SharedOutputCommitter,
    authorization: sim_runtime::AuthorizedCapabilities,
    cx: &mut gpui::App,
) -> anyhow::Result<()> {
    let binding =
        asset_binding(cx).ok_or_else(|| anyhow::anyhow!("The project reader is unavailable."))?;
    binding.update(cx, |binding, _cx| {
        binding.set_publisher(committer, authorization)
    })
}

pub fn open_authorized_png_artboard(
    choice: interop::GraphitePngAssetChoice,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> anyhow::Result<gpui::Task<anyhow::Result<gpui::Entity<GraphiteArtboardItem>>>> {
    use gpui::AppContext as _;
    anyhow::ensure!(
        !cancellation.is_cancelled(),
        "Opening the PNG was cancelled."
    );
    let binding = choice.binding()?;
    anyhow::ensure!(
        asset_binding(cx).is_some_and(|current| current.entity_id() == binding.entity_id()),
        "The project asset profile changed. Reopen the chooser."
    );
    let project = binding.update(cx, |binding, cx| binding.project_context(cx))?;
    project.track_operation(&cancellation, cx)?;
    let background = cx.background_spawn({
        let project = project.clone();
        let cancellation = cancellation.clone();
        async move { interop::acquire_linked_png(choice, project, cancellation) }
    });
    install_linked_artboard(background, project, cancellation, window, cx)
}

pub fn query_published_png_versions(
    offset: usize,
    text: Option<String>,
    cx: &mut gpui::App,
) -> anyhow::Result<gpui::Task<anyhow::Result<interop::GraphitePublishedPngPage>>> {
    interop::query_published_png_versions(offset, text, cx)
}

pub fn open_published_png_artboard(
    choice: interop::GraphitePublishedPngChoice,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> anyhow::Result<gpui::Task<anyhow::Result<gpui::Entity<GraphiteArtboardItem>>>> {
    use gpui::AppContext as _;
    anyhow::ensure!(
        !cancellation.is_cancelled(),
        "Opening the PNG was cancelled."
    );
    let binding = choice.binding()?;
    anyhow::ensure!(
        asset_binding(cx).is_some_and(|current| current.entity_id() == binding.entity_id()),
        "The project asset profile changed. Reopen the chooser."
    );
    let project = binding.update(cx, |binding, cx| binding.project_context(cx))?;
    project.track_operation(&cancellation, cx)?;
    let background = cx.background_spawn({
        let project = project.clone();
        let cancellation = cancellation.clone();
        async move { interop::acquire_published_png(choice, project, cancellation) }
    });
    install_linked_artboard(background, project, cancellation, window, cx)
}

pub fn query_saved_sim_documents(
    offset: usize,
    text: Option<String>,
    cx: &mut gpui::App,
) -> anyhow::Result<gpui::Task<anyhow::Result<interop::GraphiteSimDocumentPage>>> {
    interop::query_saved_sim_documents(offset, text, cx)
}

pub fn open_saved_sim_document(
    choice: interop::GraphiteSimDocumentChoice,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> anyhow::Result<gpui::Task<anyhow::Result<gpui::Entity<GraphiteArtboardItem>>>> {
    use gpui::AppContext as _;
    anyhow::ensure!(
        !cancellation.is_cancelled(),
        "Opening the .sim was cancelled."
    );
    let binding = choice.binding()?;
    anyhow::ensure!(
        asset_binding(cx).is_some_and(|current| current.entity_id() == binding.entity_id()),
        "The project asset profile changed. Reopen the chooser."
    );
    let project = binding.update(cx, |binding, cx| binding.project_context(cx))?;
    project.track_operation(&cancellation, cx)?;
    let background = cx.background_spawn({
        let project = project.clone();
        let cancellation = cancellation.clone();
        async move { interop::acquire_saved_sim(choice, project, cancellation) }
    });
    install_linked_artboard(background, project, cancellation, window, cx)
}

fn install_linked_artboard(
    background: gpui::Task<
        anyhow::Result<(
            graphics_core::evaluate::CpuSvgArtboardDocument,
            interop::GraphiteLinkedDocument,
        )>,
    >,
    project: interop::GraphiteProjectContext,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    window: &mut gpui::Window,
    cx: &mut gpui::Context<workspace::Workspace>,
) -> anyhow::Result<gpui::Task<anyhow::Result<gpui::Entity<GraphiteArtboardItem>>>> {
    use gpui::AppContext as _;
    Ok(cx.spawn_in(window, async move |workspace, cx| {
        let (document, linked) = background.await?;
        workspace.update_in(cx, |workspace, window, cx| {
            anyhow::ensure!(
                !cancellation.is_cancelled(),
                "Opening the document was cancelled."
            );
            anyhow::ensure!(
                project.current(cx),
                "The project asset permission changed. Reopen the chooser."
            );
            let binding = asset_binding(cx)
                .ok_or_else(|| anyhow::anyhow!("The project reader was closed."))?;
            let session =
                binding.update(cx, |binding, cx| binding.bind_linked_session(document, cx))?;
            let item = cx.new(|cx| {
                GraphiteArtboardItem::from_session(session, 128 * 1024 * 1024, window, cx)
            });
            item.update(cx, |item, cx| {
                item.set_linked_document(linked, cx);
                item.configure_png_limits(128 * 1024 * 1024, 64 * 1024 * 1024, cx);
                item.enable_automatic_preview(window, cx);
            });
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(Box::new(item.clone()), true, true, None, window, cx);
            });
            Ok(item)
        })?
    }))
}
