use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Weak},
};

use gpui::{Context, ImageSource, Render, RenderImage, Role, Window, img};
use graphics_core::resources::{
    AdmittedCpuImage, CpuImagePixels, CpuImageResult, CpuMixParameters, CpuMixResult,
};
use image::{Frame, RgbaImage};
use ui::prelude::*;

use crate::resources::{ImageResourceError, ImageResourceState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuPreviewError {
    SceneRenderRequired,
    SizeOverflow,
    PreviewBudgetExceeded,
    SourceConversionFailed,
    InvalidPixelLayout,
    ParentIdentityMismatch,
    ParentDescriptorMismatch,
    ParentNotFresh,
    ResultNotFresh,
    ResourceRevoked,
}

impl fmt::Display for CpuPreviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SceneRenderRequired => formatter.write_str("This image requires scene rendering"),
            Self::SizeOverflow => {
                formatter.write_str("The preview image dimensions exceed the supported size")
            }
            Self::PreviewBudgetExceeded => {
                formatter.write_str("The preview pixel buffer exceeds its byte limit")
            }
            Self::SourceConversionFailed => {
                formatter.write_str("The source image conversion failed")
            }
            Self::InvalidPixelLayout => {
                formatter.write_str("The converted image has an invalid pixel layout")
            }
            Self::ParentIdentityMismatch => {
                formatter.write_str("The image result does not match the committed asset version")
            }
            Self::ParentDescriptorMismatch => formatter
                .write_str("The image result does not match the committed dimensions or transform"),
            Self::ResultNotFresh => formatter.write_str("The composition result is not current"),
            Self::ResourceRevoked => {
                formatter.write_str("The image resource authorization was revoked")
            }
            Self::ParentNotFresh => {
                formatter.write_str("The image result's parent is not currently fresh")
            }
        }
    }
}

impl std::error::Error for CpuPreviewError {}

pub fn prepare_cpu_preview(
    image: &AdmittedCpuImage,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuPreviewError> {
    if image.descriptor().transform != [1., 0., 0., 1., 0., 0.] {
        return Err(CpuPreviewError::SceneRenderRequired);
    }
    prepare_original_cpu_pixels(
        image.image().width,
        image.image().height,
        max_preview_bytes,
        || image.image().to_flat_u8(),
    )
}

pub fn prepare_cpu_result_preview(
    result: &CpuImageResult,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuPreviewError> {
    prepare_cpu_pixels_preview(result.pixels(), max_preview_bytes)
}

pub fn prepare_cpu_pixels_preview(
    pixels: &CpuImagePixels,
    max_preview_bytes: usize,
) -> Result<Arc<RenderImage>, CpuPreviewError> {
    if pixels.requires_scene_render() {
        return Err(CpuPreviewError::SceneRenderRequired);
    }
    prepare_original_cpu_pixels(
        pixels.image().width,
        pixels.image().height,
        max_preview_bytes,
        || pixels.image().to_flat_u8(),
    )
}

fn prepare_original_cpu_pixels(
    width: u32,
    height: u32,
    max_preview_bytes: usize,
    convert: impl FnOnce() -> (Vec<u8>, u32, u32),
) -> Result<Arc<RenderImage>, CpuPreviewError> {
    let width = usize::try_from(width).map_err(|_| CpuPreviewError::SizeOverflow)?;
    let height = usize::try_from(height).map_err(|_| CpuPreviewError::SizeOverflow)?;
    let preview_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(CpuPreviewError::SizeOverflow)?;
    // The source conversion's pixel buffer is moved into the frame; this is not a
    // bound on the retained source image, GPUI atlas storage, or global heap use.
    if preview_bytes > max_preview_bytes {
        return Err(CpuPreviewError::PreviewBudgetExceeded);
    }
    let (mut pixels, width, height) = catch_unwind(AssertUnwindSafe(convert))
        .map_err(|_| CpuPreviewError::SourceConversionFailed)?;
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    let buffer =
        RgbaImage::from_raw(width, height, pixels).ok_or(CpuPreviewError::InvalidPixelLayout)?;
    Ok(Arc::new(RenderImage::new(vec![Frame::new(buffer)])))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CpuPreviewDiagnostic {
    Resource(ImageResourceError),
    Preparation(CpuPreviewError),
}

impl fmt::Display for CpuPreviewDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => fmt::Display::fmt(error, formatter),
            Self::Preparation(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

#[derive(Clone)]
pub struct CpuPreviewParent {
    source: Option<Arc<AdmittedCpuImage>>,
    revision: u64,
    fresh: bool,
    revoked: bool,
    diagnostic: Option<ImageResourceError>,
}

impl CpuPreviewParent {
    pub fn from_state(state: &ImageResourceState) -> Self {
        Self {
            source: state.committed().cloned(),
            revision: state.revision(),
            fresh: state.is_fresh(),
            revoked: state.is_revoked(),
            diagnostic: state.diagnostic().cloned(),
        }
    }

    fn committed(&self) -> Option<&Arc<AdmittedCpuImage>> {
        self.source.as_ref()
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn is_fresh(&self) -> bool {
        self.fresh
    }
    fn is_revoked(&self) -> bool {
        self.revoked
    }
    fn diagnostic(&self) -> Option<&ImageResourceError> {
        self.diagnostic.as_ref()
    }
}

pub struct MixPreviewRequest<'a> {
    pub parents: [CpuPreviewParent; 2],
    pub result: Option<&'a Arc<CpuMixResult>>,
    pub parameters: CpuMixParameters,
    pub current_generation: u64,
    pub result_generation: Option<u64>,
    pub result_is_fresh: bool,
}

struct PreparedMixPreview {
    under_revision: u64,
    under: Weak<AdmittedCpuImage>,
    result: Weak<CpuMixResult>,
    generation: u64,
}

struct PreparedScenePreview {
    scene: Weak<graphics_core::render::PreparedCpuSvgComposition>,
    generation: u64,
}

struct PreparedPreview {
    revision: u64,
    source: Weak<AdmittedCpuImage>,
    result: Option<Weak<CpuImageResult>>,
    mix: Option<PreparedMixPreview>,
    scene: Option<PreparedScenePreview>,
    image: Arc<RenderImage>,
}

pub struct CpuPreviewPresentation {
    prepared: Option<PreparedPreview>,
    retained_scene: Option<Arc<crate::session::PreparedImageScenePreview>>,
    retained_artboard: Option<crate::session::PreparedArtboardPreview>,
    fresh: bool,
    diagnostic: Option<CpuPreviewDiagnostic>,
}

impl CpuPreviewPresentation {
    pub fn new(cx: &mut Context<Self>) -> Self {
        cx.on_release(|presentation, cx| {
            if let Some(prepared) = presentation.prepared.take() {
                cx.drop_image(prepared.image, None);
            }
            if let Some(preview) = presentation.retained_artboard.take() {
                cx.drop_image(preview.image().clone(), None);
            }
        })
        .detach();
        Self {
            prepared: None,
            retained_scene: None,
            retained_artboard: None,
            fresh: false,
            diagnostic: None,
        }
    }

    pub fn image(&self) -> Option<&Arc<RenderImage>> {
        self.prepared
            .as_ref()
            .map(|prepared| &prepared.image)
            .or_else(|| {
                self.retained_artboard
                    .as_ref()
                    .map(|preview| preview.image())
            })
    }

    pub fn revision(&self) -> Option<u64> {
        self.prepared
            .as_ref()
            .map(|prepared| prepared.revision)
            .or_else(|| {
                self.retained_artboard
                    .as_ref()
                    .map(|preview| preview.output().revision())
            })
    }

    pub fn is_fresh(&self) -> bool {
        self.fresh
    }

    pub fn diagnostic(&self) -> Option<&CpuPreviewDiagnostic> {
        self.diagnostic.as_ref()
    }

    pub fn last_valid_qualification(&self) -> Option<&'static str> {
        (self.image().is_some() && !self.fresh).then_some("Last valid image")
    }

    pub fn accept_artboard_preview(
        &mut self,
        session: &crate::session::GraphiteArtboardSession,
        preview: &crate::session::PreparedArtboardPreview,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        if session.resource_is_revoked() {
            self.refresh_artboard_resource_status(session, window, cx);
            return Err(CpuPreviewError::ResourceRevoked);
        }
        if !session.preview_is_current(preview) {
            let error = CpuPreviewError::ResultNotFresh;
            self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
            return Err(error);
        }
        if self.retained_artboard.as_ref().is_some_and(|retained| {
            retained.generation() == preview.generation()
                && Arc::ptr_eq(retained.image(), preview.image())
                && Arc::ptr_eq(
                    &retained.output().output_lease(),
                    &preview.output().output_lease(),
                )
        }) {
            self.set_status(true, None, cx);
            return Ok(());
        }
        self.clear_preview(window, cx);
        self.retained_artboard = Some(preview.clone());
        self.set_status(true, None, cx);
        cx.notify();
        Ok(())
    }

    pub fn refresh_artboard_status(
        &mut self,
        session: &crate::session::GraphiteArtboardSession,
        cx: &mut Context<Self>,
    ) {
        let fresh = self
            .retained_artboard
            .as_ref()
            .is_some_and(|preview| session.preview_is_current(preview));
        let diagnostic = (!fresh).then_some(CpuPreviewDiagnostic::Preparation(
            CpuPreviewError::ResultNotFresh,
        ));
        self.set_status(fresh, diagnostic, cx);
    }

    pub fn refresh_artboard_resource_status(
        &mut self,
        session: &crate::session::GraphiteArtboardSession,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if session.resource_is_revoked() {
            self.clear_preview(window, cx);
            self.set_status(
                false,
                Some(CpuPreviewDiagnostic::Preparation(
                    CpuPreviewError::ResourceRevoked,
                )),
                cx,
            );
        } else {
            self.refresh_artboard_status(session, cx);
        }
    }

    pub fn refresh(
        &mut self,
        state: &ImageResourceState,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        let diagnostic = state
            .diagnostic()
            .cloned()
            .map(CpuPreviewDiagnostic::Resource);
        let source = match (state.is_revoked(), state.committed()) {
            (false, Some(source)) => source,
            _ => {
                self.clear_preview(window, cx);
                let diagnostic = diagnostic.or_else(|| {
                    state
                        .is_revoked()
                        .then_some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
                });
                self.set_status(false, diagnostic, cx);
                return Ok(());
            }
        };
        let source_reference = Arc::downgrade(source);
        if self.prepared.as_ref().is_some_and(|prepared| {
            prepared.revision == state.revision()
                && prepared.result.is_none()
                && prepared.mix.is_none()
                && prepared.scene.is_none()
                && Weak::ptr_eq(&prepared.source, &source_reference)
        }) {
            self.set_status(state.is_fresh(), diagnostic, cx);
            return Ok(());
        }
        let image = match prepare_cpu_preview(source, max_preview_bytes) {
            Ok(image) => image,
            Err(error) => {
                self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
                return Err(error);
            }
        };
        self.replace_preview(
            PreparedPreview {
                revision: state.revision(),
                source: source_reference,
                result: None,
                mix: None,
                scene: None,
                image,
            },
            window,
            cx,
        );
        self.set_status(state.is_fresh(), diagnostic, cx);
        Ok(())
    }

    pub fn refresh_result(
        &mut self,
        state: &ImageResourceState,
        result: &Arc<CpuImageResult>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        let diagnostic = state
            .diagnostic()
            .cloned()
            .map(CpuPreviewDiagnostic::Resource);
        let source = match (state.is_revoked(), state.committed()) {
            (false, Some(source)) => source,
            _ => {
                self.clear_preview(window, cx);
                let diagnostic = diagnostic.or_else(|| {
                    state
                        .is_revoked()
                        .then_some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
                });
                self.set_status(false, diagnostic, cx);
                return Ok(());
            }
        };
        if !state.is_fresh() {
            self.set_status(false, diagnostic, cx);
            return Err(CpuPreviewError::ParentNotFresh);
        }
        if result.input_identity() != source.identity() {
            let error = CpuPreviewError::ParentIdentityMismatch;
            self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
            return Err(error);
        }
        let parent_descriptor = source.descriptor();
        let result_descriptor = result.descriptor();
        if result.input_origin() != source.origin()
            || parent_descriptor.width != result_descriptor.width
            || parent_descriptor.height != result_descriptor.height
            || parent_descriptor.transform.map(f64::to_bits)
                != result_descriptor.transform.map(f64::to_bits)
        {
            let error = CpuPreviewError::ParentDescriptorMismatch;
            self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
            return Err(error);
        }
        let source_reference = Arc::downgrade(source);
        let result_reference = Arc::downgrade(result);
        if self.prepared.as_ref().is_some_and(|prepared| {
            prepared.revision == state.revision()
                && prepared.scene.is_none()
                && prepared.mix.is_none()
                && Weak::ptr_eq(&prepared.source, &source_reference)
                && prepared
                    .result
                    .as_ref()
                    .is_some_and(|prepared_result| Weak::ptr_eq(prepared_result, &result_reference))
        }) {
            self.set_status(true, diagnostic, cx);
            return Ok(());
        }
        let image = match prepare_cpu_result_preview(result, max_preview_bytes) {
            Ok(image) => image,
            Err(error) => {
                self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
                return Err(error);
            }
        };
        self.replace_preview(
            PreparedPreview {
                revision: state.revision(),
                source: source_reference,
                result: Some(result_reference),
                mix: None,
                scene: None,
                image,
            },
            window,
            cx,
        );
        self.set_status(true, diagnostic, cx);
        Ok(())
    }

    pub(crate) fn accept_scene_preview(
        &mut self,
        session: &crate::session::GraphiteImageSession,
        preview: crate::session::PreparedImageScenePreview,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        if !session.scene_preview_is_current(&preview) {
            return Err(CpuPreviewError::ResultNotFresh);
        }
        self.refresh_scene(session, &preview, window, cx)?;
        self.retained_scene = Some(Arc::new(preview));
        Ok(())
    }

    pub(crate) fn retained_scene_preview(
        &self,
    ) -> Option<&Arc<crate::session::PreparedImageScenePreview>> {
        self.retained_scene.as_ref()
    }

    pub(crate) fn refresh_selected_scene(
        &mut self,
        session: &crate::session::GraphiteImageSession,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<Result<(), CpuPreviewError>> {
        let preview = self.retained_scene_preview()?.clone();
        Some(self.refresh_scene(session, &preview, window, cx))
    }

    pub(crate) fn refresh_scene(
        &mut self,
        session: &crate::session::GraphiteImageSession,
        preview: &crate::session::PreparedImageScenePreview,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        let state = session.resource_state();
        let diagnostic = state
            .diagnostic()
            .cloned()
            .map(CpuPreviewDiagnostic::Resource);
        if state.is_revoked() || state.committed().is_none() {
            self.clear_preview(window, cx);
            self.set_status(
                false,
                diagnostic.or_else(|| {
                    state
                        .is_revoked()
                        .then_some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
                }),
                cx,
            );
            return Ok(());
        }
        if !session.scene_preview_is_current(preview) {
            let error = if state.is_fresh() {
                CpuPreviewError::ResultNotFresh
            } else {
                CpuPreviewError::ParentNotFresh
            };
            self.set_status(
                false,
                diagnostic.or(Some(CpuPreviewDiagnostic::Preparation(error))),
                cx,
            );
            return Err(error);
        }
        let source = Arc::downgrade(&preview.source);
        let result = preview.result.as_ref().map(Arc::downgrade);
        let scene = Arc::downgrade(&preview.scene);
        if self.prepared.as_ref().is_some_and(|prepared| {
            prepared.revision == preview.source_revision
                && Weak::ptr_eq(&prepared.source, &source)
                && prepared.mix.is_none()
                && match (prepared.result.as_ref(), result.as_ref()) {
                    (None, None) => true,
                    (Some(prepared), Some(result)) => Weak::ptr_eq(prepared, result),
                    _ => false,
                }
                && prepared.scene.as_ref().is_some_and(|prepared| {
                    prepared.generation == preview.scene_generation
                        && Weak::ptr_eq(&prepared.scene, &scene)
                })
                && Arc::ptr_eq(&prepared.image, &preview.image)
        }) {
            self.set_status(true, diagnostic, cx);
            return Ok(());
        }
        self.replace_preview(
            PreparedPreview {
                revision: preview.source_revision,
                source,
                result,
                mix: None,
                scene: Some(PreparedScenePreview {
                    scene,
                    generation: preview.scene_generation,
                }),
                image: preview.image.clone(),
            },
            window,
            cx,
        );
        self.set_status(true, diagnostic, cx);
        Ok(())
    }

    pub fn refresh_mix(
        &mut self,
        request: MixPreviewRequest<'_>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), CpuPreviewError> {
        let [over, under] = &request.parents;
        let diagnostic = [over, under]
            .into_iter()
            .find_map(|parent| parent.diagnostic().cloned())
            .map(CpuPreviewDiagnostic::Resource);
        if over.is_revoked() || under.is_revoked() {
            self.clear_preview(window, cx);
            self.set_status(
                false,
                Some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked)),
                cx,
            );
            return Ok(());
        }
        let [Some(over_source), Some(under_source)] = [over.committed(), under.committed()] else {
            self.clear_preview(window, cx);
            self.set_status(false, diagnostic, cx);
            return Ok(());
        };
        let failure = if !over.is_fresh() || !under.is_fresh() {
            Some(CpuPreviewError::ParentNotFresh)
        } else if !request.result_is_fresh
            || request.result_generation != Some(request.current_generation)
        {
            Some(CpuPreviewError::ResultNotFresh)
        } else {
            None
        };
        if let Some(error) = failure {
            self.set_status(
                false,
                diagnostic.or(Some(CpuPreviewDiagnostic::Preparation(error))),
                cx,
            );
            return Err(error);
        }
        let Some(result) = request.result else {
            let error = CpuPreviewError::ResultNotFresh;
            self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
            return Err(error);
        };
        if result.parameters() != request.parameters {
            let error = CpuPreviewError::ResultNotFresh;
            self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
            return Err(error);
        }
        for (parent, source) in result.parents().iter().zip([over_source, under_source]) {
            let error = if parent.identity() != source.identity() {
                Some(CpuPreviewError::ParentIdentityMismatch)
            } else if parent.origin() != source.origin()
                || parent.descriptor() != source.descriptor()
                || parent.descriptor().transform.map(f64::to_bits)
                    != source.descriptor().transform.map(f64::to_bits)
            {
                Some(CpuPreviewError::ParentDescriptorMismatch)
            } else {
                None
            };
            if let Some(error) = error {
                self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
                return Err(error);
            }
        }
        let source_reference = Arc::downgrade(over_source);
        let under_reference = Arc::downgrade(under_source);
        let result_reference = Arc::downgrade(result);
        if self.prepared.as_ref().is_some_and(|prepared| {
            prepared.revision == over.revision()
                && prepared.scene.is_none()
                && prepared.result.is_none()
                && Weak::ptr_eq(&prepared.source, &source_reference)
                && prepared.mix.as_ref().is_some_and(|mix| {
                    mix.under_revision == under.revision()
                        && mix.generation == request.current_generation
                        && Weak::ptr_eq(&mix.under, &under_reference)
                        && Weak::ptr_eq(&mix.result, &result_reference)
                })
        }) {
            self.set_status(true, diagnostic, cx);
            return Ok(());
        }
        let image = match prepare_cpu_pixels_preview(result.pixels(), max_preview_bytes) {
            Ok(image) => image,
            Err(error) => {
                self.set_status(false, Some(CpuPreviewDiagnostic::Preparation(error)), cx);
                return Err(error);
            }
        };
        self.replace_preview(
            PreparedPreview {
                revision: over.revision(),
                source: source_reference,
                result: None,
                mix: Some(PreparedMixPreview {
                    under_revision: under.revision(),
                    under: under_reference,
                    result: result_reference,
                    generation: request.current_generation,
                }),
                scene: None,
                image,
            },
            window,
            cx,
        );
        self.set_status(true, diagnostic, cx);
        Ok(())
    }

    fn clear_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.retained_scene = None;
        if let Some(preview) = self.retained_artboard.take() {
            cx.drop_image(preview.image().clone(), Some(window));
            cx.notify();
        }
        if let Some(previous) = self.prepared.take() {
            cx.drop_image(previous.image, Some(window));
            cx.notify();
        }
    }

    fn replace_preview(
        &mut self,
        prepared: PreparedPreview,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.retained_scene = None;
        if let Some(preview) = self.retained_artboard.take() {
            cx.drop_image(preview.image().clone(), Some(window));
            cx.notify();
        }
        if let Some(previous) = self.prepared.replace(prepared) {
            cx.drop_image(previous.image, Some(window));
        }
        cx.notify();
    }

    fn set_status(
        &mut self,
        fresh: bool,
        diagnostic: Option<CpuPreviewDiagnostic>,
        cx: &mut Context<Self>,
    ) {
        if self.fresh != fresh || self.diagnostic != diagnostic {
            self.fresh = fresh;
            self.diagnostic = diagnostic;
            cx.notify();
        }
    }
}

impl Render for CpuPreviewPresentation {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("graphite-cpu-preview")
            .role(Role::Group)
            .aria_label("Image preview")
            .size_full()
            .gap_1()
            .when_some(self.image().cloned(), |view, image| {
                view.child(
                    div()
                        .id("graphite-cpu-preview-image")
                        .role(Role::Image)
                        .aria_label("Committed image")
                        .debug_selector(|| "GRAPHITE-CPU-PREVIEW-IMAGE".into())
                        .w_full()
                        .flex_1()
                        .min_h(px(0.))
                        .child(img(ImageSource::Render(image)).size_full()),
                )
            })
            .when_some(self.last_valid_qualification(), |view, qualification| {
                view.child(
                    div()
                        .id("graphite-cpu-preview-last-valid")
                        .debug_selector(|| "GRAPHITE-CPU-PREVIEW-LAST-VALID".into())
                        .child(Label::new(qualification)),
                )
            })
            .when_some(self.diagnostic(), |view, diagnostic| {
                view.child(
                    div()
                        .id("graphite-cpu-preview-diagnostic")
                        .role(Role::Alert)
                        .debug_selector(|| "GRAPHITE-CPU-PREVIEW-DIAGNOSTIC".into())
                        .child(Label::new(diagnostic.to_string()).color(Color::Error)),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{ImageRequestDecision, ImageResourceFailure, ImageResourceRequest};
    use gpui::{DevicePixels, Entity, TestAppContext, VisualTestContext, point, size};
    use graphics_core::resources::{
        AlphaAssociation, CpuImageBudget, CpuImageDescriptor, CpuImageEdit, CpuSampleFormat,
        ImageColorSpace, ImageResourceIdentity, admit_cpu_image, evaluate_cpu_image_edit,
    };

    const PIXELS: [u8; 16] = [
        255, 0, 0, 255, 0, 0, 255, 255, 17, 83, 201, 0, 0, 255, 0, 128,
    ];
    const PIXELS_SHA256: [u8; 32] = [
        162, 141, 9, 211, 10, 128, 239, 240, 173, 10, 13, 139, 212, 21, 160, 39, 75, 105, 248, 30,
        202, 213, 39, 212, 203, 236, 76, 166, 168, 14, 56, 157,
    ];

    fn descriptor(width: u32, height: u32, row_stride: usize) -> CpuImageDescriptor {
        CpuImageDescriptor {
            width,
            height,
            row_stride,
            sample_format: CpuSampleFormat::Rgba8,
            color_space: ImageColorSpace::Srgb,
            alpha: AlphaAssociation::Straight,
            transform: [1., 0., 0., 1., 0., 0.],
        }
    }

    fn admitted(
        bytes: &[u8],
        content_sha256: [u8; 32],
        descriptor: CpuImageDescriptor,
    ) -> AdmittedCpuImage {
        admit_cpu_image(
            ImageResourceIdentity {
                asset_id: "project/preview-image".into(),
                version_id: "immutable-version-1".into(),
                content_sha256,
            },
            descriptor,
            Arc::from(bytes),
            CpuImageBudget {
                max_input_bytes: bytes.len(),
                max_decoded_bytes: 4096,
            },
        )
        .expect("the exact fixture is admitted")
    }

    #[test]
    fn at_graphite_cpu_preview_uses_original_pixels_in_bgra_order() {
        let input = admitted(&PIXELS, PIXELS_SHA256, descriptor(2, 2, 8));
        let preview = prepare_cpu_preview(&input, 16).expect("identity image preview");
        assert_eq!(preview.frame_count(), 1);
        assert_eq!(preview.size(0), size(DevicePixels(2), DevicePixels(2)));
        assert_eq!(
            preview.as_bytes(0),
            Some([0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 128].as_slice())
        );
        assert_eq!(preview.as_bytes(1), None);
        let (mut original, width, height) = input.image().to_flat_u8();
        for pixel in original.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        assert_eq!((width, height), (2, 2));
        assert_eq!(preview.as_bytes(0), Some(original.as_slice()));
    }

    #[test]
    fn at_graphite_cpu_preview_retains_source_alpha_ramp_and_transparency() {
        let pixels = [
            90, 40, 200, 0, 90, 40, 200, 1, 90, 40, 200, 64, 90, 40, 200, 128, 90, 40, 200, 254,
            90, 40, 200, 255,
        ];
        let digest = [
            250, 85, 56, 156, 129, 180, 74, 116, 24, 144, 229, 205, 86, 36, 192, 123, 174, 149, 28,
            133, 142, 60, 174, 246, 201, 169, 19, 156, 126, 128, 235, 5,
        ];
        let input = admitted(&pixels, digest, descriptor(6, 1, 24));
        let preview = prepare_cpu_preview(&input, 24).expect("alpha ramp preview");
        assert_eq!(
            preview.as_bytes(0),
            Some(
                [
                    0, 0, 0, 0, 200, 40, 90, 1, 200, 40, 90, 64, 200, 40, 90, 128, 200, 40, 90,
                    254, 200, 40, 90, 255,
                ]
                .as_slice()
            )
        );
    }

    #[test]
    fn at_graphite_cpu_preview_excludes_wire_row_padding() {
        let pixels = [
            255, 0, 0, 255, 0, 0, 255, 255, 11, 12, 13, 14, 0, 255, 0, 128, 17, 83, 201, 0, 15, 16,
            17, 18,
        ];
        let digest = [
            36, 10, 187, 251, 243, 92, 103, 243, 169, 203, 235, 23, 85, 179, 143, 159, 166, 22,
            221, 193, 176, 235, 153, 200, 37, 125, 143, 200, 211, 150, 58, 169,
        ];
        let input = admitted(&pixels, digest, descriptor(2, 2, 12));
        let preview = prepare_cpu_preview(&input, 16).expect("padded rows preview");
        assert_eq!(preview.size(0), size(DevicePixels(2), DevicePixels(2)));
        assert_eq!(
            preview.as_bytes(0),
            Some([0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 0, 0].as_slice())
        );
    }

    #[test]
    fn at_graphite_cpu_preview_rejects_transforms_without_mutating_source() {
        for transform in [
            [1., 0., 0., 1., 2., 0.],
            [2., 0., 0., 1., 0., 0.],
            [0., 1., -1., 0., 0., 0.],
        ] {
            let mut descriptor = descriptor(2, 2, 8);
            descriptor.transform = transform;
            let input = admitted(&PIXELS, PIXELS_SHA256, descriptor);
            let original = input.image().clone();
            assert_eq!(
                prepare_cpu_preview(&input, 16),
                Err(CpuPreviewError::SceneRenderRequired)
            );
            assert_eq!(input.image(), &original);
            assert_eq!(input.descriptor(), &descriptor);
        }
    }

    #[test]
    fn at_graphite_cpu_preview_repeat_preserves_identity_descriptor_and_source() {
        let input = admitted(&PIXELS, PIXELS_SHA256, descriptor(2, 2, 8));
        let identity = input.identity().clone();
        let descriptor = *input.descriptor();
        let original = input.image().clone();
        let original_base64 = input.image().base64_string.clone();
        let first = prepare_cpu_preview(&input, 16).expect("first preview");
        let second = prepare_cpu_preview(&input, 16).expect("repeat preview");
        assert_eq!(first.as_bytes(0), second.as_bytes(0));
        assert_eq!(first.size(0), second.size(0));
        assert_ne!(first.id, second.id);
        assert_eq!(input.identity(), &identity);
        assert_eq!(input.descriptor(), &descriptor);
        assert_eq!(input.image(), &original);
        assert_eq!(input.image().base64_string, original_base64);
        drop(input);
        assert_eq!(
            first.as_bytes(0),
            Some([0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 255, 0, 128].as_slice())
        );
        assert_eq!(first.as_bytes(0), second.as_bytes(0));
    }

    #[test]
    fn at_graphite_cpu_preview_budget_rejection_is_atomic_and_recovers() {
        let input = admitted(&PIXELS, PIXELS_SHA256, descriptor(2, 2, 8));
        let identity = input.identity().clone();
        let descriptor = *input.descriptor();
        let original = input.image().clone();
        let original_base64 = input.image().base64_string.clone();
        let exact = prepare_cpu_preview(&input, 16).expect("the exact pixel buffer cap is enough");

        for limit in [15, 0] {
            assert_eq!(
                prepare_cpu_preview(&input, limit),
                Err(CpuPreviewError::PreviewBudgetExceeded)
            );
            assert_eq!(input.identity(), &identity);
            assert_eq!(input.descriptor(), &descriptor);
            assert_eq!(input.image(), &original);
            assert_eq!(input.image().base64_string, original_base64);
        }

        let recovered = prepare_cpu_preview(&input, 16).expect("retry with the exact cap succeeds");
        assert_eq!(recovered.as_bytes(0), exact.as_bytes(0));
        assert_eq!(recovered.size(0), exact.size(0));
        assert_eq!(input.identity(), &identity);
        assert_eq!(input.descriptor(), &descriptor);
        assert_eq!(input.image(), &original);
        assert_eq!(input.image().base64_string, original_base64);
    }

    fn load_version(
        state: &mut ImageResourceState,
        version: &str,
        descriptor: CpuImageDescriptor,
    ) -> (ImageResourceRequest, AdmittedCpuImage) {
        let identity = ImageResourceIdentity {
            asset_id: "project/presentation-image".into(),
            version_id: version.into(),
            content_sha256: PIXELS_SHA256,
        };
        let ImageRequestDecision::Load(request) = state
            .request(identity.clone(), descriptor)
            .expect("new immutable version request")
        else {
            panic!("this control requires a fresh request");
        };
        let image = admit_cpu_image(
            identity,
            descriptor,
            Arc::from(PIXELS),
            CpuImageBudget {
                max_input_bytes: PIXELS.len(),
                max_decoded_bytes: 4096,
            },
        )
        .expect("original source fixture admission");
        (request, image)
    }

    fn commit_version(state: &mut ImageResourceState, version: &str) {
        let (request, image) = load_version(state, version, descriptor(2, 2, 8));
        state
            .accept(&request, image)
            .expect("matching current completion");
    }

    fn presented_image(
        owner: &Entity<CpuPreviewPresentation>,
        window: &mut VisualTestContext,
    ) -> Arc<RenderImage> {
        window.update(|_, cx| {
            owner
                .read(cx)
                .image()
                .expect("prepared presentation")
                .clone()
        })
    }

    fn paint_preview(window: &mut VisualTestContext, image: &Arc<RenderImage>) {
        window.draw(point(px(0.), px(0.)), size(px(32.), px(32.)), |_, _| {
            img(ImageSource::Render(image.clone()))
                .size_full()
                .into_any_element()
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(image)));
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_reuses_one_image_for_committed_revision(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("initial preview preparation");
        let first = presented_image(&owner, window);
        paint_preview(window, &first);
        for _ in 0..3 {
            window
                .update(|window, cx| {
                    owner.update(cx, |owner, cx| owner.refresh(&state, 0, window, cx))
                })
                .expect("cached refresh does not allocate another preview buffer");
            let repeated = presented_image(&owner, window);
            assert!(Arc::ptr_eq(&first, &repeated));
            assert_eq!(first.id, repeated.id);
        }
        window.update(|_, cx| {
            let owner = owner.read(cx);
            assert_eq!(owner.revision(), Some(state.revision()));
            assert!(owner.is_fresh());
            assert_eq!(owner.last_valid_qualification(), None);
            assert_eq!(owner.diagnostic(), None);
        });
        assert!(window.update(|window, _| window.has_image_atlas_entry(&first)));
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_replacement_retires_atlas_in_both_windows(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("first revision");
        let previous = presented_image(&owner, first_window);
        paint_preview(first_window, &previous);
        paint_preview(second_window, &previous);

        commit_version(&mut state, "second");
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("second revision");
        let current = presented_image(&owner, first_window);
        assert_ne!(previous.id, current.id);
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&previous)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&previous)));
        paint_preview(first_window, &current);
        paint_preview(second_window, &current);
        first_window.update(|_, cx| {
            assert_eq!(owner.read(cx).revision(), Some(2));
            assert!(owner.read(cx).is_fresh());
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_same_revision_in_another_state_replaces_pixels(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut first_state = ImageResourceState::default();
        let mut second_state = ImageResourceState::default();
        commit_version(&mut first_state, "first");
        commit_version(&mut second_state, "second");
        assert_eq!(first_state.revision(), second_state.revision());
        assert!(!Arc::ptr_eq(
            first_state.committed().expect("first source"),
            second_state.committed().expect("second source")
        ));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&first_state, 16, window, cx))
            })
            .expect("first state presentation");
        let previous = presented_image(&owner, window);
        paint_preview(window, &previous);
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&second_state, 0, window, cx))
            }),
            Err(CpuPreviewError::PreviewBudgetExceeded)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &previous));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&second_state, 16, window, cx))
            })
            .expect("another state's source needs its own preparation");
        assert_ne!(presented_image(&owner, window).id, previous.id);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        window.update(|_, cx| {
            assert_eq!(owner.read(cx).revision(), Some(1));
            assert!(owner.read(cx).is_fresh());
            assert_eq!(owner.read(cx).diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_failed_preparation_preserves_last_valid_and_recovers(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("first prepared revision");
        let previous = presented_image(&owner, window);
        paint_preview(window, &previous);
        commit_version(&mut state, "second");
        let committed = state.committed().expect("new verified source").clone();
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 15, window, cx))
            }),
            Err(CpuPreviewError::PreviewBudgetExceeded)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &previous));
        assert!(window.update(|window, _| window.has_image_atlas_entry(&previous)));
        window.update(|_, cx| {
            let owner = owner.read(cx);
            assert_eq!(owner.revision(), Some(1));
            assert!(!owner.is_fresh());
            assert_eq!(owner.last_valid_qualification(), Some("Last valid image"));
            assert_eq!(
                owner.diagnostic(),
                Some(&CpuPreviewDiagnostic::Preparation(
                    CpuPreviewError::PreviewBudgetExceeded
                ))
            );
        });
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert_eq!(state.diagnostic(), None);
        assert!(Arc::ptr_eq(
            state.committed().expect("source retained"),
            &committed
        ));

        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("recovered preparation");
        assert_ne!(presented_image(&owner, window).id, previous.id);
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        window.update(|_, cx| {
            assert!(owner.read(cx).is_fresh());
            assert_eq!(owner.read(cx).diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_pending_and_obsolete_results_remain_last_valid(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("first presentation");
        let previous = presented_image(&owner, window);
        let (obsolete, obsolete_image) = load_version(&mut state, "obsolete", descriptor(2, 2, 8));
        let (active, _) = load_version(&mut state, "active", descriptor(2, 2, 8));
        assert_eq!(
            state.accept(&obsolete, obsolete_image),
            Err(ImageResourceError::StaleRequest)
        );
        assert_eq!(state.diagnostic(), None);
        window
            .update(|window, cx| owner.update(cx, |owner, cx| owner.refresh(&state, 0, window, cx)))
            .expect("pending refresh retains prepared image");
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &previous));
        window.update(|_, cx| {
            assert!(!owner.read(cx).is_fresh());
            assert_eq!(
                owner.read(cx).last_valid_qualification(),
                Some("Last valid image")
            );
        });
        assert_eq!(
            state.reject(&active, ImageResourceFailure::Denied),
            Err(ImageResourceError::Denied)
        );
        window
            .update(|window, cx| owner.update(cx, |owner, cx| owner.refresh(&state, 0, window, cx)))
            .expect("denied completion preserves last valid pixels");
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &previous));
        window.update(|_, cx| {
            assert_eq!(
                owner.read(cx).diagnostic(),
                Some(&CpuPreviewDiagnostic::Resource(ImageResourceError::Denied))
            );
            assert_eq!(owner.read(cx).revision(), Some(1));
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_revocation_stays_hidden_across_new_request_and_failure(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("initial visible preview");
        let previous = presented_image(&owner, window);
        paint_preview(window, &previous);
        let identity = state
            .committed()
            .expect("committed identity")
            .identity()
            .clone();
        state
            .revoke(&identity)
            .expect("committed resource revocation");
        let (request, _) = load_version(&mut state, "pending", descriptor(2, 2, 8));
        assert_eq!(state.diagnostic(), None);
        assert!(state.is_revoked());
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("revoked preview retirement");
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        window.update(|_, cx| {
            assert!(owner.read(cx).image().is_none());
            assert_eq!(owner.read(cx).last_valid_qualification(), None);
            assert_eq!(
                owner.read(cx).diagnostic(),
                Some(&CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
            );
        });
        assert_eq!(
            state.reject(&request, ImageResourceFailure::Cancelled),
            Err(ImageResourceError::Cancelled)
        );
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("failed revoked refresh remains hidden");
        window.update(|_, cx| assert!(owner.read(cx).image().is_none()));
        commit_version(&mut state, "fresh-authorized-completion");
        assert!(!state.is_revoked());
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("fresh validated source can be presented");
        assert_ne!(presented_image(&owner, window).id, previous.id);
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_split_owner_final_release_retires_both_atlases(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view_owner = owner.clone();
        let second_view_owner = owner.clone();
        let weak_owner = owner.downgrade();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("shared presentation");
        let image = presented_image(&owner, first_window);
        paint_preview(first_window, &image);
        paint_preview(second_window, &image);
        drop(owner);
        drop(first_view_owner);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_some());
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert_eq!(
            presented_image(&second_view_owner, second_window).id,
            image.id
        );
        drop(second_view_owner);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
    }

    #[gpui::test]
    fn at_graphite_cpu_presentation_release_during_window_update_retires_current_atlas(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let weak_owner = owner.downgrade();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("presentation before current-window release");
        let image = presented_image(&owner, first_window);
        paint_preview(first_window, &image);
        paint_preview(second_window, &image);
        first_window.update(|window, _| {
            drop(owner);
            assert!(window.has_image_atlas_entry(&image));
        });
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
    }

    fn init_preview_view(cx: &mut TestAppContext) {
        cx.update(|cx| {
            assets::Assets.load_test_fonts(cx);
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
    }

    fn paint_presentation(window: &mut VisualTestContext, owner: &Entity<CpuPreviewPresentation>) {
        window.draw(point(px(0.), px(0.)), size(px(320.), px(240.)), |_, _| {
            owner.clone().into_any_element()
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_preview_view_draws_stale_error_and_recovered_entity(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("visible committed revision");
        let previous = presented_image(&owner, window);
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(window.update(|window, _| window.has_image_atlas_entry(&previous)));

        let (pending, _) = load_version(&mut state, "pending", descriptor(2, 2, 8));
        window
            .update(|window, cx| owner.update(cx, |owner, cx| owner.refresh(&state, 0, window, cx)))
            .expect("last valid presentation during pending request");
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert_eq!(
            state.reject(&pending, ImageResourceFailure::Denied),
            Err(ImageResourceError::Denied)
        );
        window
            .update(|window, cx| owner.update(cx, |owner, cx| owner.refresh(&state, 0, window, cx)))
            .expect("resource failure preserves qualified pixels");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &previous));

        commit_version(&mut state, "second");
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 15, window, cx))
            }),
            Err(CpuPreviewError::PreviewBudgetExceeded)
        );
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(window.update(|window, _| window.has_image_atlas_entry(&previous)));
        window.update(|_, cx| {
            assert_eq!(
                owner.read(cx).diagnostic(),
                Some(&CpuPreviewDiagnostic::Preparation(
                    CpuPreviewError::PreviewBudgetExceeded
                ))
            );
        });
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("recovered view");
        let current = presented_image(&owner, window);
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        assert!(window.update(|window, _| window.has_image_atlas_entry(&current)));
    }

    #[gpui::test]
    fn at_graphite_cpu_preview_view_revocation_hides_pixels_until_valid_completion(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("visible preview");
        let previous = presented_image(&owner, window);
        paint_presentation(window, &owner);
        state
            .revoke(
                &state
                    .committed()
                    .expect("committed source")
                    .identity()
                    .clone(),
            )
            .expect("committed source revocation");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("hide revoked source");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        let (request, _) = load_version(&mut state, "pending", descriptor(2, 2, 8));
        assert_eq!(
            state.reject(&request, ImageResourceFailure::Cancelled),
            Err(ImageResourceError::Cancelled)
        );
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("new request failure does not unhide revoked source");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        commit_version(&mut state, "second");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("validated recovery");
        paint_presentation(window, &owner);
        let current = presented_image(&owner, window);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(window.update(|window, _| window.has_image_atlas_entry(&current)));
    }

    #[gpui::test]
    fn at_graphite_cpu_preview_view_shared_entity_retires_both_window_atlases(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view_owner = owner.clone();
        let second_view_owner = owner.clone();
        let weak_owner = owner.downgrade();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "first");
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("shared view preparation");
        let image = presented_image(&owner, first_window);
        paint_presentation(first_window, &first_view_owner);
        paint_presentation(second_window, &second_view_owner);
        assert!(
            first_window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE")
                .is_some()
        );
        assert!(
            second_window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE")
                .is_some()
        );
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        drop(owner);
        drop(first_view_owner);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_some());
        paint_presentation(second_window, &second_view_owner);
        assert_eq!(
            presented_image(&second_view_owner, second_window).id,
            image.id
        );
        second_window.update(|_, _| drop(second_view_owner));
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
    }

    #[test]
    fn at_graphite_cpu_result_preview_uses_original_edit_rgb_alpha_and_immutable_float_result() {
        let input = admitted(&PIXELS, PIXELS_SHA256, descriptor(2, 2, 8));
        let original_source = input.image().clone();
        let identity = input.identity().clone();
        let source_descriptor = *input.descriptor();
        for (operation, expected) in [
            (
                CpuImageEdit::Invert,
                [
                    255, 255, 0, 255, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 255, 128,
                ],
            ),
            (
                CpuImageEdit::MakeOpaque,
                [
                    0, 0, 255, 255, 255, 0, 0, 255, 201, 83, 17, 255, 0, 255, 0, 255,
                ],
            ),
        ] {
            let result = evaluate_cpu_image_edit(&input, operation, 4096)
                .expect("actual original image edit");
            let image = result.image().clone();
            let attributes = result.attributes().clone();
            let digest = result.pixels_sha256();
            let result_descriptor = *result.descriptor();
            let source_pointer = std::ptr::from_ref(result.image());
            let float_bits: Vec<_> = result
                .image()
                .data
                .iter()
                .map(|pixel| [pixel.r(), pixel.g(), pixel.b(), pixel.a()].map(f32::to_bits))
                .collect();
            assert!(!result.requires_scene_render());
            assert_eq!(result.input_identity(), &identity);
            assert_eq!(result.operation(), operation);
            assert_eq!(
                result_descriptor.sample_format,
                CpuSampleFormat::Rgba32Float
            );
            assert_eq!(result_descriptor.color_space, ImageColorSpace::LinearSrgb);
            assert_eq!(result_descriptor.alpha, AlphaAssociation::Straight);
            let first = prepare_cpu_result_preview(&result, 16).expect("original result preview");
            let second = prepare_cpu_result_preview(&result, 16).expect("repeat result preview");
            assert_eq!(first.frame_count(), 1);
            assert_eq!(first.size(0), size(DevicePixels(2), DevicePixels(2)));
            assert_eq!(first.as_bytes(0), Some(expected.as_slice()));
            assert_eq!(second.as_bytes(0), first.as_bytes(0));
            let (mut original, width, height) = result.image().to_flat_u8();
            for pixel in original.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
            assert_eq!((width, height), (2, 2));
            assert_eq!(first.as_bytes(0), Some(original.as_slice()));
            assert_eq!(result.image(), &image);
            assert_eq!(std::ptr::from_ref(result.image()), source_pointer);
            assert_eq!(
                result
                    .image()
                    .data
                    .iter()
                    .map(|pixel| [pixel.r(), pixel.g(), pixel.b(), pixel.a()].map(f32::to_bits))
                    .collect::<Vec<_>>(),
                float_bits
            );
            assert_eq!(result.attributes(), &attributes);
            assert_eq!(result.pixels_sha256(), digest);
            assert_eq!(result.descriptor(), &result_descriptor);
            assert_eq!(input.image(), &original_source);
            assert_eq!(input.identity(), &identity);
            assert_eq!(input.descriptor(), &source_descriptor);
            drop(result);
            assert_eq!(first.as_bytes(0), Some(expected.as_slice()));
        }
    }

    #[test]
    fn at_graphite_cpu_result_preview_budget_failure_is_atomic_and_recovers() {
        let input = admitted(&PIXELS, PIXELS_SHA256, descriptor(2, 2, 8));
        let source = input.image().clone();
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            let result = evaluate_cpu_image_edit(&input, operation, 4096)
                .expect("original image edit before preview admission");
            let image = result.image().clone();
            let attributes = result.attributes().clone();
            let digest = result.pixels_sha256();
            let identity = result.input_identity().clone();
            let descriptor = *result.descriptor();
            for cap in [0, 15] {
                assert_eq!(
                    prepare_cpu_result_preview(&result, cap).err(),
                    Some(CpuPreviewError::PreviewBudgetExceeded)
                );
                assert_eq!(result.image(), &image);
                assert_eq!(result.attributes(), &attributes);
                assert_eq!(result.pixels_sha256(), digest);
                assert_eq!(result.input_identity(), &identity);
                assert_eq!(result.descriptor(), &descriptor);
                assert_eq!(result.operation(), operation);
            }
            let recovered = prepare_cpu_result_preview(&result, 16)
                .expect("exact preview buffer cap admits recovery");
            assert_eq!(recovered.size(0), size(DevicePixels(2), DevicePixels(2)));
            assert_eq!(recovered.as_bytes(0).expect("prepared bytes").len(), 16);
            assert_eq!(result.image(), &image);
            assert_eq!(result.pixels_sha256(), digest);
            assert_eq!(input.image(), &source);
        }
    }

    #[test]
    fn at_graphite_cpu_result_preview_rejects_scene_transforms_before_pixel_conversion() {
        for transform in [
            [1., 0., 0., 1., 2., 0.],
            [2., 0., 0., 1., 0., 0.],
            [1., 0.5, 0., 1., 0., 0.],
        ] {
            let input = admitted(
                &PIXELS,
                PIXELS_SHA256,
                CpuImageDescriptor {
                    transform,
                    ..descriptor(2, 2, 8)
                },
            );
            let source = input.image().clone();
            for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
                let result = evaluate_cpu_image_edit(&input, operation, 4096)
                    .expect("source edit retains typed transform");
                let image = result.image().clone();
                let attributes = result.attributes().clone();
                let digest = result.pixels_sha256();
                assert!(result.requires_scene_render());
                for cap in [0, 16] {
                    assert_eq!(
                        prepare_cpu_result_preview(&result, cap).err(),
                        Some(CpuPreviewError::SceneRenderRequired)
                    );
                }
                assert_eq!(result.descriptor().transform, transform);
                assert_eq!(result.image(), &image);
                assert_eq!(result.attributes(), &attributes);
                assert_eq!(result.pixels_sha256(), digest);
                assert_eq!(result.input_identity(), input.identity());
                assert_eq!(result.operation(), operation);
                assert_eq!(input.image(), &source);
                assert_eq!(input.descriptor().transform, transform);
            }
        }
    }
    fn edited_result(state: &ImageResourceState, operation: CpuImageEdit) -> Arc<CpuImageResult> {
        Arc::new(
            evaluate_cpu_image_edit(
                state.committed().expect("committed parent"),
                operation,
                4096,
            )
            .expect("actual original CPU edit"),
        )
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_distinguishes_source_operation_and_exact_arc(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "parent");
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("original source preview");
        let source_preview = presented_image(&owner, window);
        paint_preview(window, &source_preview);
        let invert = edited_result(&state, CpuImageEdit::Invert);
        let result_strong_count = Arc::strong_count(&invert);
        let parent_strong_count = Arc::strong_count(state.committed().expect("parent"));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &invert, 16, window, cx)
                })
            })
            .expect("derived result replaces source in the same owner");
        let first = presented_image(&owner, window);
        assert_ne!(first.id, source_preview.id);
        assert_ne!(first.as_bytes(0), source_preview.as_bytes(0));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&source_preview)));
        assert_eq!(Arc::strong_count(&invert), result_strong_count);
        assert_eq!(
            Arc::strong_count(state.committed().expect("parent")),
            parent_strong_count
        );
        for _ in 0..3 {
            window
                .update(|window, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_result(&state, &invert, 0, window, cx)
                    })
                })
                .expect("exact result Arc reuses prepared pixels");
            assert!(Arc::ptr_eq(&first, &presented_image(&owner, window)));
        }
        let same_operation_new_result = edited_result(&state, CpuImageEdit::Invert);
        assert!(!Arc::ptr_eq(&invert, &same_operation_new_result));
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &same_operation_new_result, 0, window, cx)
                })
            }),
            Err(CpuPreviewError::PreviewBudgetExceeded)
        );
        assert!(Arc::ptr_eq(&first, &presented_image(&owner, window)));
        window.update(|_, cx| {
            assert!(!owner.read(cx).is_fresh());
            assert_eq!(
                owner.read(cx).last_valid_qualification(),
                Some("Last valid image")
            );
        });
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &same_operation_new_result, 16, window, cx)
                })
            })
            .expect("new result object gets a new conversion");
        assert_ne!(presented_image(&owner, window).id, first.id);
        let opaque = edited_result(&state, CpuImageEdit::MakeOpaque);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &opaque, 16, window, cx)
                })
            })
            .expect("changed operation is not the prior cached result");
        let opaque_preview = presented_image(&owner, window);
        assert_ne!(opaque_preview.as_bytes(0), first.as_bytes(0));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| owner.refresh(&state, 16, window, cx))
            })
            .expect("source refresh cannot reuse derived result pixels");
        let restored_source = presented_image(&owner, window);
        assert_ne!(restored_source.id, opaque_preview.id);
        assert_eq!(restored_source.as_bytes(0), source_preview.as_bytes(0));
        window.update(|_, cx| {
            assert!(owner.read(cx).is_fresh());
            assert_eq!(owner.read(cx).revision(), Some(state.revision()));
            assert_eq!(owner.read(cx).diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_rejects_foreign_and_stale_parents(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "parent");
        let result = edited_result(&state, CpuImageEdit::Invert);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            })
            .expect("initial result");
        let previous = presented_image(&owner, window);
        let mut foreign_state = ImageResourceState::default();
        commit_version(&mut foreign_state, "foreign-version");
        let foreign = edited_result(&foreign_state, CpuImageEdit::MakeOpaque);
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &foreign, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::ParentIdentityMismatch)
        );
        assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        let (request, _) = load_version(&mut state, "pending", descriptor(2, 2, 8));
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::ParentNotFresh)
        );
        assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
        assert_eq!(
            state.reject(&request, ImageResourceFailure::Denied),
            Err(ImageResourceError::Denied)
        );
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::ParentNotFresh)
        );
        window.update(|_, cx| {
            assert_eq!(
                owner.read(cx).diagnostic(),
                Some(&CpuPreviewDiagnostic::Resource(ImageResourceError::Denied))
            );
            assert!(!owner.read(cx).is_fresh());
        });
        commit_version(&mut state, "replacement");
        let current_parent = state.committed().expect("new parent").clone();
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::ParentIdentityMismatch)
        );
        assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
        let current = edited_result(&state, CpuImageEdit::Invert);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &current, 16, window, cx)
                })
            })
            .expect("exact new parent/result recovers");
        assert_ne!(presented_image(&owner, window).id, previous.id);
        assert!(Arc::ptr_eq(
            &current_parent,
            state.committed().expect("parent not mutated")
        ));
        assert_eq!(state.revision(), 2);
        assert!(state.is_fresh());
        assert_eq!(state.diagnostic(), None);
        window.update(|_, cx| {
            assert!(owner.read(cx).is_fresh());
            assert_eq!(owner.read(cx).diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_requires_exact_parent_transform_and_dimensions(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "parent");
        let result = edited_result(&state, CpuImageEdit::Invert);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            })
            .expect("valid result");
        let previous = presented_image(&owner, window);
        for descriptor in [
            CpuImageDescriptor {
                transform: [1., 0., 0., 1., 10., 0.],
                ..descriptor(2, 2, 8)
            },
            CpuImageDescriptor {
                transform: [1., -0.0, 0., 1., 0., 0.],
                ..descriptor(2, 2, 8)
            },
            descriptor(4, 1, 16),
        ] {
            let mut different_parent = ImageResourceState::default();
            let (request, image) = load_version(&mut different_parent, "parent", descriptor);
            different_parent
                .accept(&request, image)
                .expect("same immutable wire rows, different descriptor");
            let mismatched = edited_result(&different_parent, CpuImageEdit::Invert);
            assert_eq!(
                mismatched.input_identity(),
                state.committed().expect("parent").identity()
            );
            assert_eq!(
                window.update(|window, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_result(&state, &mismatched, 0, window, cx)
                    })
                }),
                Err(CpuPreviewError::ParentDescriptorMismatch)
            );
            assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
        }
        let mut transformed_state = ImageResourceState::default();
        let (request, image) = load_version(
            &mut transformed_state,
            "scene",
            CpuImageDescriptor {
                transform: [1., 0., 0., 1., 10., 0.],
                ..descriptor(2, 2, 8)
            },
        );
        transformed_state
            .accept(&request, image)
            .expect("transformed source admission");
        let scene_result = edited_result(&transformed_state, CpuImageEdit::MakeOpaque);
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&transformed_state, &scene_result, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::SceneRenderRequired)
        );
        assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_same_revision_other_parent_reprepares(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut first_state = ImageResourceState::default();
        let mut second_state = ImageResourceState::default();
        commit_version(&mut first_state, "same-immutable-parent");
        commit_version(&mut second_state, "same-immutable-parent");
        assert_eq!(first_state.revision(), second_state.revision());
        let result = edited_result(&first_state, CpuImageEdit::Invert);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&first_state, &result, 16, window, cx)
                })
            })
            .expect("first owner");
        let previous = presented_image(&owner, window);
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&second_state, &result, 0, window, cx)
                })
            }),
            Err(CpuPreviewError::PreviewBudgetExceeded)
        );
        assert!(Arc::ptr_eq(&previous, &presented_image(&owner, window)));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&second_state, &result, 16, window, cx)
                })
            })
            .expect("second parent's Arc owns a separate preparation key");
        assert_ne!(presented_image(&owner, window).id, previous.id);
        let weak_result = Arc::downgrade(&result);
        drop(result);
        assert!(weak_result.upgrade().is_none());
        assert!(window.update(|_, cx| owner.read(cx).image().is_some()));
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_revocation_hides_until_fresh_validation(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "parent");
        let result = edited_result(&state, CpuImageEdit::Invert);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            })
            .expect("visible result");
        let previous = presented_image(&owner, window);
        paint_presentation(window, &owner);
        assert!(window.update(|window, _| window.has_image_atlas_entry(&previous)));
        let identity = state
            .committed()
            .expect("committed source")
            .identity()
            .clone();
        state.revoke(&identity).expect("parent revocation");
        let (request, _) = load_version(&mut state, "pending", descriptor(2, 2, 8));
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            })
            .expect("persistent revoked input retires result");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&previous)));
        assert_eq!(
            state.reject(&request, ImageResourceFailure::Cancelled),
            Err(ImageResourceError::Cancelled)
        );
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            })
            .expect("failed request cannot clear revocation");
        window.update(|_, cx| {
            assert!(owner.read(cx).image().is_none());
            assert!(!owner.read(cx).is_fresh());
        });
        commit_version(&mut state, "new-validated-parent");
        assert_eq!(
            window.update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &result, 16, window, cx)
                })
            }),
            Err(CpuPreviewError::ParentIdentityMismatch)
        );
        assert!(window.update(|_, cx| owner.read(cx).image().is_none()));
        let current = edited_result(&state, CpuImageEdit::MakeOpaque);
        window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &current, 16, window, cx)
                })
            })
            .expect("fresh exact parent and new edit can reappear");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert_ne!(presented_image(&owner, window).id, previous.id);
    }

    #[gpui::test]
    fn at_graphite_cpu_result_presentation_shared_windows_replace_and_final_release(
        cx: &mut TestAppContext,
    ) {
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view_owner = owner.clone();
        let second_view_owner = owner.clone();
        let weak_owner = owner.downgrade();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        let mut state = ImageResourceState::default();
        commit_version(&mut state, "parent");
        let first_result = edited_result(&state, CpuImageEdit::Invert);
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &first_result, 16, window, cx)
                })
            })
            .expect("shared result preparation");
        let previous = presented_image(&owner, first_window);
        paint_preview(first_window, &previous);
        paint_preview(second_window, &previous);
        let second_result = edited_result(&state, CpuImageEdit::MakeOpaque);
        first_window
            .update(|window, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_result(&state, &second_result, 16, window, cx)
                })
            })
            .expect("changed edit retires both previous atlas entries");
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&previous)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&previous)));
        let current = presented_image(&owner, first_window);
        paint_preview(first_window, &current);
        paint_preview(second_window, &current);
        drop(owner);
        drop(first_view_owner);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_some());
        assert_eq!(
            presented_image(&second_view_owner, second_window).id,
            current.id
        );
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&current)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&current)));
        second_window.update(|_, _| drop(second_view_owner));
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&current)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&current)));
    }
    fn authenticated_scene_session(
        cx: &mut TestAppContext,
        transform: [f64; 6],
    ) -> (
        crate::resources::tests::HostReaderFixture,
        crate::resources::tests::IndexedRequestPng,
        Entity<crate::session::GraphiteImageSession>,
    ) {
        let fixture = crate::resources::tests::host_reader_fixture()
            .expect("real indexed service and Input grant");
        let png = crate::resources::tests::indexed_request_png(&fixture, 4)
            .expect("actual tensor PNG producer");
        let source = cx.new(|_| crate::session::GraphiteImageSession::new(4096));
        source
            .update(cx, |source, cx| {
                source.load_encoded_png(
                    crate::session::GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: png.asset_identity.clone(),
                        authorization: fixture.input_authorization.clone(),
                        identity: png.image_identity.clone(),
                        transform,
                        encoded_budget: graphics_core::resources::EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                    },
                    cx,
                )
            })
            .expect("authenticated original-network PNG load");
        cx.run_until_parked();
        source.update(cx, |source, cx| {
            source.set_operation(Some(graphics_core::resources::CpuImageEdit::Invert), cx)
        });
        cx.run_until_parked();
        assert!(source.read_with(cx, |source, _| source.result_is_fresh()));
        (fixture, png, source)
    }

    fn presentation_scene_configuration() -> graphics_core::render::CpuSvgSceneConfiguration {
        graphics_core::render::CpuSvgSceneConfiguration {
            viewport: graphite_core_types::transform::Footprint {
                transform: graphite_glam::DAffine2::IDENTITY,
                resolution: graphite_glam::UVec2::new(16, 8),
                quality: graphite_core_types::transform::Footprint::DEFAULT.quality,
            },
            scale: 1.,
            max_svg_bytes: 32 * 1024,
        }
    }

    #[gpui::test]
    async fn at_r20_2_scene_presentation_accepts_exact_pixels_and_separates_cpu_cache(
        cx: &mut TestAppContext,
    ) {
        let (_fixture, _png, source) = authenticated_scene_session(cx, [1., 0., 0., 1., 0., 0.]);
        let owner = cx.new(CpuPreviewPresentation::new);
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("existing tracked scene job")
            .await
            .expect("original scene conversion");
        let expected = cx
            .update(|cx| cx.svg_renderer())
            .render_single_frame(frame.scene.svg().svg.as_bytes(), 1.)
            .expect("existing SVG renderer binding oracle");
        assert_eq!(frame.image.as_bytes(0), expected.as_bytes(0));
        let before_pixels = frame
            .result
            .as_ref()
            .expect("edited scene receipt")
            .pixels()
            .image()
            .clone();
        let before_attributes = frame
            .result
            .as_ref()
            .expect("edited scene receipt")
            .pixels()
            .attributes()
            .clone();
        let window = cx.add_empty_window();
        for _ in 0..3 {
            window
                .update(|window, cx| {
                    source.update(cx, |source, cx| {
                        owner.update(cx, |owner, cx| {
                            owner.refresh_scene(source, &frame, window, cx)
                        })
                    })
                })
                .expect("synchronous current scene acceptance");
            assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
            assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        }
        paint_preview(window, &frame.image);
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_result(
                            source.resource_state(),
                            frame.result.as_ref().expect("edited scene receipt"),
                            4096,
                            window,
                            cx,
                        )
                    })
                })
            })
            .expect("pixel-only result path cannot reuse scene cache");
        let pixels = presented_image(&owner, window);
        assert!(!Arc::ptr_eq(&pixels, &frame.image));
        assert_eq!((pixels.size(0).width.0, pixels.size(0).height.0), (3, 1));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        paint_preview(window, &pixels);
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("same valid receipt replaces CPU presentation");
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&pixels)));
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh(source.resource_state(), 4096, window, cx)
                    })
                })
            })
            .expect("original source refresh cannot reuse scene result");
        let original = presented_image(&owner, window);
        assert!(!Arc::ptr_eq(&original, &frame.image));
        assert_eq!(
            original.as_bytes(0),
            prepare_cpu_preview(&frame.source, 4096)
                .expect("source pixel oracle")
                .as_bytes(0)
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .pixels()
                .image(),
            &before_pixels
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .pixels()
                .attributes(),
            &before_attributes
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .input_identity(),
            frame.source.identity()
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .input_origin(),
            frame.source.origin()
        );
    }

    #[gpui::test]
    async fn at_r20_2_scene_presentation_rejects_new_generation_and_recovers_after_failure(
        cx: &mut TestAppContext,
    ) {
        let (fixture, png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        let owner = cx.new(CpuPreviewPresentation::new);
        let first = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("first scene request")
            .await
            .expect("actual converted scene");
        let window = cx.add_empty_window();
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &first, window, cx)
                    })
                })
            })
            .expect("initial current frame");
        paint_preview(window, &first.image);
        let pending = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2047, cx)
            })
            .expect("successor scene job");
        assert_eq!(
            window.update(|window, cx| source.update(cx, |source, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_scene(source, &first, window, cx)
                })
            })),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &first.image));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        assert!(
            pending
                .await
                .err()
                .expect("actual cap failure")
                .contains("byte limit")
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &first.image));
        let recovered = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 0.5, 512, cx)
            })
            .expect("explicit scene recovery")
            .await
            .expect("new scale conversion");
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &recovered, window, cx)
                    })
                })
            })
            .expect("current generation publication");
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &recovered.image
        ));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&first.image)));
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            None
        );
        let pending = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("pending cancel control");
        source.update(window, |source, cx| source.cancel(cx));
        assert!(pending.await.is_err());
        assert_eq!(
            window.update(|window, cx| source.update(cx, |source, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_scene(source, &recovered, window, cx)
                })
            })),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &recovered.image
        ));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        let mut identity = png.image_identity.clone();
        identity.version_id = "explicit-test-scene-load-successor".into();
        source
            .update(window, |source, cx| {
                source.load_encoded_png(
                    crate::session::GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: png.asset_identity.clone(),
                        authorization: fixture.input_authorization.clone(),
                        identity,
                        transform: [16., 0., 0., 8., 0., 0.],
                        encoded_budget: graphics_core::resources::EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                    },
                    cx,
                )
            })
            .expect("actual authenticated source-version successor");
        assert!(!source.read_with(window, |source, _| source.resource_state().is_fresh()));
        assert_eq!(
            window.update(|window, cx| source.update(cx, |source, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_scene(source, &recovered, window, cx)
                })
            })),
            Err(CpuPreviewError::ParentNotFresh)
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &recovered.image
        ));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        window.run_until_parked();
        let replacement = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("new committed source scene")
            .await
            .expect("source successor scene conversion");
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &replacement, window, cx)
                    })
                })
            })
            .expect("fresh source recovery");
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &replacement.image
        ));
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert!(replacement.source_revision > recovered.source_revision);
        assert_ne!(
            replacement.source.identity().version_id,
            recovered.source.identity().version_id
        );
    }

    #[gpui::test]
    async fn at_r20_2_scene_presentation_revocation_missing_and_foreign_owner_are_not_published(
        cx: &mut TestAppContext,
    ) {
        let (_fixture, _png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        let (_other_fixture, _other_png, foreign) =
            authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        let missing = cx.new(|_| crate::session::GraphiteImageSession::new(4096));
        let owner = cx.new(CpuPreviewPresentation::new);
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("current scene request")
            .await
            .expect("original conversion");
        let window = cx.add_empty_window();
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("current publication");
        paint_preview(window, &frame.image);
        assert_eq!(
            window.update(|window, cx| foreign.update(cx, |foreign, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_scene(foreign, &frame, window, cx)
                })
            })),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("legitimate owner recovery");
        paint_preview(window, &frame.image);
        window
            .update(|window, cx| {
                missing.update(cx, |missing, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(missing, &frame, window, cx)
                    })
                })
            })
            .expect("missing parent clears an actually presented scene");
        assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("legitimate owner restores current scene after missing-input rejection");
        paint_preview(window, &frame.image);
        source
            .update(window, |source, cx| {
                source.revoke(frame.source.identity(), cx)
            })
            .expect("real committed resource revocation");
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("revoked scene clears existing image");
        assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            Some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        for _ in 0..2 {
            window
                .update(|window, cx| {
                    source.update(cx, |source, cx| {
                        owner.update(cx, |owner, cx| {
                            owner.refresh_scene(source, &frame, window, cx)
                        })
                    })
                })
                .expect("revoked receipt remains hidden");
            assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
        }
        window
            .update(|window, cx| {
                missing.update(cx, |missing, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(missing, &frame, window, cx)
                    })
                })
            })
            .expect("missing parent cannot publish receipt");
        assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
    }

    #[gpui::test]
    async fn at_r20_2_scene_presentation_shared_windows_use_weak_scene_and_retire_atlases(
        cx: &mut TestAppContext,
    ) {
        let (_fixture, _png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view = owner.clone();
        let second_view = owner.clone();
        let weak_owner = owner.downgrade();
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("shared scene request")
            .await
            .expect("actual original conversion");
        let weak_scene = Arc::downgrade(&frame.scene);
        let weak_input = Arc::downgrade(&frame.source);
        let weak_result = Arc::downgrade(frame.result.as_ref().expect("edited scene receipt"));
        let weak_session = source.downgrade();
        let image = frame.image.clone();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        first_window
            .update(|first_window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, first_window, cx)
                    })
                })
            })
            .expect("shared current scene");
        paint_preview(first_window, &image);
        paint_preview(second_window, &image);
        second_window
            .update(|second_window, cx| {
                source.update(cx, |source, cx| {
                    second_view.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, second_window, cx)
                    })
                })
            })
            .expect("split refresh reuses exact converted image");
        assert!(Arc::ptr_eq(
            &presented_image(&first_view, first_window),
            &image
        ));
        assert!(Arc::ptr_eq(
            &presented_image(&second_view, second_window),
            &image
        ));
        drop(frame);
        assert!(
            weak_scene.upgrade().is_none(),
            "presentation retains a weak scene reference"
        );
        drop(source);
        first_window.cx.update(|_| ());
        assert!(weak_session.upgrade().is_none());
        assert!(weak_input.upgrade().is_none());
        assert!(weak_result.upgrade().is_none());
        drop(owner);
        drop(first_view);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_some());
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        drop(second_view);
        first_window.cx.update(|_| ());
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
    }
    #[gpui::test]
    async fn at_r20_2_scene_render_view_shows_fresh_stale_and_revoked_presentation(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let (_fixture, _png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        let owner = cx.new(CpuPreviewPresentation::new);
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("existing source-owned scene task")
            .await
            .expect("actual original scene conversion");
        let scene_bytes = frame
            .image
            .as_bytes(0)
            .expect("expected single scene frame")
            .to_vec();
        let before_source = frame.source.image().clone();
        let before_result = frame
            .result
            .as_ref()
            .expect("edited scene receipt")
            .pixels()
            .image()
            .clone();
        let before_attributes = frame
            .result
            .as_ref()
            .expect("edited scene receipt")
            .pixels()
            .attributes()
            .clone();
        let window = cx.add_empty_window();
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("current scene publication");
        paint_presentation(window, &owner);
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        assert_eq!(
            presented_image(&owner, window)
                .as_bytes(0)
                .expect("expected presented scene frame"),
            scene_bytes
        );
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(window.update(|window, _| window.has_image_atlas_entry(&frame.image)));

        let failed = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2047, cx)
            })
            .expect("new tracked scene generation");
        assert_eq!(
            window.update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            }),
            Err(CpuPreviewError::ResultNotFresh)
        );
        paint_presentation(window, &owner);
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        assert_eq!(
            presented_image(&owner, window)
                .as_bytes(0)
                .expect("expected presented scene frame"),
            scene_bytes
        );
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            Some(CpuPreviewDiagnostic::Preparation(
                CpuPreviewError::ResultNotFresh
            ))
        );
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        assert!(
            failed.await.is_err(),
            "failed preparation cannot replace the drawn scene"
        );

        source
            .update(window, |source, cx| {
                source.revoke(frame.source.identity(), cx)
            })
            .expect("genuine parent revocation");
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("revocation clears existing scene owner");
        paint_presentation(window, &owner);
        assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            Some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
        );
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        assert_eq!(frame.source.image(), &before_source);
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .pixels()
                .image(),
            &before_result
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .pixels()
                .attributes(),
            &before_attributes
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .input_identity(),
            frame.source.identity()
        );
        assert_eq!(
            frame
                .result
                .as_ref()
                .expect("edited scene receipt")
                .input_origin(),
            frame.source.origin()
        );
        assert_eq!(
            frame
                .image
                .as_bytes(0)
                .expect("retained single scene frame"),
            scene_bytes
        );
    }

    #[gpui::test]
    async fn at_r20_2_original_scene_cache_reuses_none_result_and_separates_edited_frames(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let (fixture, png, source) = authenticated_scene_session(cx, [1., 0., 0., 1., 0., 0.]);
        source.update(cx, |source, cx| source.set_operation(None, cx));
        let owner = cx.new(CpuPreviewPresentation::new);
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("Original scene request")
            .await
            .expect("actual original scene");
        assert!(frame.result.is_none());
        let source_before = (
            frame.source.image().clone(),
            *frame.source.descriptor(),
            frame.source.identity().clone(),
            frame.source.origin(),
        );
        let window = cx.add_empty_window();
        for _ in 0..3 {
            window
                .update(|window, cx| {
                    source.update(cx, |source, cx| {
                        owner.update(cx, |owner, cx| {
                            owner.refresh_scene(source, &frame, window, cx)
                        })
                    })
                })
                .expect("repeated Original receipt");
            paint_presentation(window, &owner);
            assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
            assert_eq!(
                presented_image(&owner, window).as_bytes(0),
                frame.image.as_bytes(0)
            );
            owner.read_with(window, |owner, _| {
                assert!(owner.is_fresh());
                let cached = owner.prepared.as_ref().expect("cached source scene");
                assert!(cached.result.is_none());
                assert!(cached.scene.is_some());
                assert!(cached.mix.is_none());
            });
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        }
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh(source.resource_state(), 4096, window, cx)
                    })
                })
            })
            .expect("pixel source refresh must not reuse scene frame");
        let pixels = presented_image(&owner, window);
        assert!(!Arc::ptr_eq(&pixels, &frame.image));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        paint_presentation(window, &owner);
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("current scene restores source frame");
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&pixels)));
        paint_presentation(window, &owner);
        source.update(window, |source, cx| {
            source.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        assert_eq!(
            window.update(|window, cx| source
                .update(cx, |source, cx| owner.update(cx, |owner, cx| owner
                    .refresh_scene(source, &frame, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        window.run_until_parked();
        let edited = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("edited scene request")
            .await
            .expect("edited scene frame");
        assert!(edited.result.is_some());
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &edited, window, cx)
                    })
                })
            })
            .expect("edited scene replaces Original cache");
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &edited.image));
        assert!(!Arc::ptr_eq(&edited.image, &frame.image));
        assert_ne!(edited.image.as_bytes(0), frame.image.as_bytes(0));
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        paint_presentation(window, &owner);
        source.update(window, |source, cx| source.set_operation(None, cx));
        assert_eq!(
            window.update(|window, cx| source
                .update(cx, |source, cx| owner.update(cx, |owner, cx| owner
                    .refresh_scene(source, &edited, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        let restored = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("restored Original request")
            .await
            .expect("restored Original scene");
        assert!(restored.result.is_none());
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &restored, window, cx)
                    })
                })
            })
            .expect("Original replaces edited cache");
        assert_eq!(
            presented_image(&owner, window).as_bytes(0),
            frame.image.as_bytes(0)
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&edited.image)));
        assert_eq!(
            (
                frame.source.image().clone(),
                *frame.source.descriptor(),
                frame.source.identity().clone(),
                frame.source.origin()
            ),
            source_before
        );

        paint_presentation(window, &owner);
        source
            .update(window, |source, cx| {
                source.load_encoded_png(
                    crate::session::GraphiteEncodedPngLoad {
                        assets: fixture.assets.clone(),
                        asset_identity: png.asset_identity.clone(),
                        authorization: fixture.input_authorization.clone(),
                        identity: png.image_identity.clone(),
                        transform: [16., 0., 0., 8., 0., 0.],
                        encoded_budget: graphics_core::resources::EncodedPngBudget {
                            max_encoded_bytes: 16 * 1024,
                            max_pixel_bytes: 4096,
                            max_metadata_bytes: 4096,
                        },
                    },
                    cx,
                )
            })
            .expect("same authenticated input with explicit scene transform");
        window.run_until_parked();
        assert!(!source.read_with(window, |source, _| {
            source.scene_preview_is_current(&restored)
        }));
        assert_eq!(
            window.update(|window, cx| source
                .update(cx, |source, cx| owner.update(cx, |owner, cx| owner
                    .refresh(source.resource_state(), 4096, window, cx)))),
            Err(CpuPreviewError::SceneRenderRequired)
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &restored.image
        ));
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            Some(CpuPreviewDiagnostic::Preparation(
                CpuPreviewError::SceneRenderRequired
            ))
        );
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        let transformed = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("transformed Original scene request")
            .await
            .expect("actual scene recovery");
        assert!(transformed.result.is_none());
        assert_eq!(
            transformed.source.descriptor().transform,
            [16., 0., 0., 8., 0., 0.]
        );
        assert_eq!(transformed.source.identity(), frame.source.identity());
        assert_eq!(transformed.source.image(), frame.source.image());
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &transformed, window, cx)
                    })
                })
            })
            .expect("scene route recovers transformed input");
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            &transformed.image
        ));
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            None
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&restored.image)));
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert_eq!(
            (
                frame.source.image().clone(),
                *frame.source.descriptor(),
                frame.source.identity().clone(),
                frame.source.origin()
            ),
            source_before
        );
    }

    #[gpui::test]
    async fn at_r20_2_original_scene_render_view_keeps_last_valid_and_hides_revocation(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let (_fixture, _png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        source.update(cx, |source, cx| source.set_operation(None, cx));
        let owner = cx.new(CpuPreviewPresentation::new);
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("Original request")
            .await
            .expect("original scene frame");
        let window = cx.add_empty_window();
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &frame, window, cx)
                    })
                })
            })
            .expect("fresh Original presentation");
        paint_presentation(window, &owner);
        let failing = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2047, cx)
            })
            .expect("next bounded generation");
        assert_eq!(
            window.update(|window, cx| source
                .update(cx, |source, cx| owner.update(cx, |owner, cx| owner
                    .refresh_scene(source, &frame, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(failing.await.is_err());
        paint_presentation(window, &owner);
        assert!(Arc::ptr_eq(&presented_image(&owner, window), &frame.image));
        assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        assert_eq!(
            owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
            Some(CpuPreviewDiagnostic::Preparation(
                CpuPreviewError::ResultNotFresh
            ))
        );
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        let recovered = source
            .update(window, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("source recovery request")
            .await
            .expect("fresh recovery");
        window
            .update(|window, cx| {
                source.update(cx, |source, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.refresh_scene(source, &recovered, window, cx)
                    })
                })
            })
            .expect("recovered source receipt");
        paint_presentation(window, &owner);
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(&frame.image)));
        source
            .update(window, |source, cx| {
                source.revoke(recovered.source.identity(), cx)
            })
            .expect("actual source revocation");
        for _ in 0..2 {
            window
                .update(|window, cx| {
                    source.update(cx, |source, cx| {
                        owner.update(cx, |owner, cx| {
                            owner.refresh_scene(source, &recovered, window, cx)
                        })
                    })
                })
                .expect("revoked Original receipt clears pixels");
            paint_presentation(window, &owner);
            assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
            assert!(!owner.read_with(window, |owner, _| owner.is_fresh()));
            assert_eq!(
                owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
                Some(CpuPreviewDiagnostic::Resource(ImageResourceError::Revoked))
            );
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                    .is_some()
            );
            assert!(!window.update(|window, _| window.has_image_atlas_entry(&recovered.image)));
        }
    }

    #[gpui::test]
    async fn at_r20_2_original_scene_shared_views_release_weak_sources_and_both_atlases(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let (_fixture, _png, source) = authenticated_scene_session(cx, [16., 0., 0., 8., 0., 0.]);
        source.update(cx, |source, cx| source.set_operation(None, cx));
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view = owner.clone();
        let second_view = owner.clone();
        let weak_owner = owner.downgrade();
        let frame = source
            .update(cx, |source, cx| {
                source.prepare_scene_preview(presentation_scene_configuration(), 1., 2048, cx)
            })
            .expect("shared Original scene request")
            .await
            .expect("Original scene frame");
        assert!(frame.result.is_none());
        let weak_input = Arc::downgrade(&frame.source);
        let weak_scene = Arc::downgrade(&frame.scene);
        let weak_session = source.downgrade();
        let image = frame.image.clone();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        for (view, window) in [
            (&first_view, &mut *first_window),
            (&second_view, &mut *second_window),
        ] {
            window
                .update(|window, cx| {
                    source.update(cx, |source, cx| {
                        view.update(cx, |owner, cx| {
                            owner.refresh_scene(source, &frame, window, cx)
                        })
                    })
                })
                .expect("shared Original receipt publication");
            paint_preview(window, &image);
            assert!(Arc::ptr_eq(&presented_image(view, window), &image));
            assert!(window.update(|window, _| window.has_image_atlas_entry(&image)));
        }
        drop(frame);
        assert!(weak_scene.upgrade().is_none());
        drop(source);
        first_window.cx.update(|_| {});
        assert!(weak_session.upgrade().is_none());
        assert!(weak_input.upgrade().is_none());
        drop(owner);
        drop(first_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_some());
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        drop(second_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(&image)));
        assert!(
            image
                .as_bytes(0)
                .expect("retained converted pixels")
                .iter()
                .any(|byte| *byte != 0)
        );
    }
    fn artboard_session_document() -> graphics_core::evaluate::CpuSvgArtboardDocument {
        use graphics_core::evaluate::{
            CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgEvaluationBudget,
        };
        use graphite_graph_craft::document::{NodeId, value::DVec2};
        use graphite_std::application_io::{ExportFormat, RenderConfig, TimingInformation};
        use graphite_std::{Color, transform::Footprint, vector_types::vector::style::RenderMode};
        CpuSvgArtboardDocument::new(
            91,
            NodeId(7001),
            CpuSvgArtboardParameters {
                location: DVec2::new(2., 2.),
                dimensions: DVec2::new(6., 4.),
                background: Color::from_gamma_srgb_channels(1., 0., 0., 0.5),
                clip: true,
            },
            RenderConfig {
                viewport: Footprint {
                    transform: graphite_glam::DAffine2::from_translation(DVec2::ONE),
                    resolution: graphite_glam::UVec2::new(31, 17),
                    quality: Footprint::DEFAULT.quality,
                },
                scale: 1.,
                time: TimingInformation {
                    time: 0.,
                    animation_time: std::time::Duration::ZERO,
                },
                pointer: DVec2::ZERO,
                render_mode: RenderMode::Normal,
                export_format: ExportFormat::Svg,
                for_export: true,
                for_eyedropper: false,
            },
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
        )
        .expect("original scoped Artboard with explicit supplied context")
    }

    #[gpui::test]
    async fn at_r20_2_artboard_presentation_rejects_foreign_stale_and_cancelled_receipts_and_recovers(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let session =
            cx.new(|_| crate::session::GraphiteArtboardSession::new(artboard_session_document()));
        let foreign =
            cx.new(|_| crate::session::GraphiteArtboardSession::new(artboard_session_document()));
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let first = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("tracked request")
            .await
            .expect("original frame");
        let original_output = first.output().output().clone();
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &first, window, cx)
                    })
                })
            })
            .expect("current publication");
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(window.update(|window, _| window.has_image_atlas_entry(first.image())));
        assert_eq!(
            window.update(|window, cx| foreign
                .update(cx, |foreign, cx| owner.update(cx, |owner, cx| owner
                    .accept_artboard_preview(foreign, &first, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), first.image()));
        assert_eq!(
            owner.read_with(window, |owner, _| owner.last_valid_qualification()),
            Some("Last valid image")
        );
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &first, window, cx)
                    })
                })
            })
            .expect("same frame revalidated without allocation");
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        let mut parameters = session.read_with(window, |session, _| session.parameters());
        parameters.dimensions.x = 9.;
        let configuration = session.read_with(window, |session, _| session.configuration());
        session
            .update(window, |session, cx| {
                session.apply(session.revision(), parameters, configuration, cx)
            })
            .expect("source successor");
        assert_eq!(
            window.update(|window, cx| session
                .update(cx, |session, cx| owner.update(cx, |owner, cx| owner
                    .accept_artboard_preview(session, &first, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        let failed = session
            .update(window, |session, cx| session.request_preview(0, cx))
            .expect("zero cap request")
            .await;
        assert!(matches!(
            failed,
            Err(crate::session::ArtboardSessionError::Conversion(_))
        ));
        session.update(window, |session, cx| {
            owner.update(cx, |owner, cx| owner.refresh_artboard_status(session, cx))
        });
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), first.image()));
        let recovered = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("recovery request")
            .await
            .expect("recovered original frame");
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &recovered, window, cx)
                    })
                })
            })
            .expect("genuine successor publication");
        assert!(!window.update(|window, _| window.has_image_atlas_entry(first.image())));
        assert_ne!(first.image().as_bytes(0), recovered.image().as_bytes(0));
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        session.update(window, |session, cx| assert!(session.cancel(cx)));
        assert_eq!(
            window.update(|window, cx| session
                .update(cx, |session, cx| owner.update(cx, |owner, cx| owner
                    .accept_artboard_preview(session, &recovered, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            recovered.image()
        ));
        assert_eq!(first.output().output(), &original_output);
    }

    #[gpui::test]
    async fn at_r20_2_artboard_shared_presentation_retires_both_atlases_on_replacement_and_release(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let session =
            cx.new(|_| crate::session::GraphiteArtboardSession::new(artboard_session_document()));
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view = owner.clone();
        let second_view = owner.clone();
        let weak_owner = owner.downgrade();
        let frame = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("tracked original request")
            .await
            .expect("genuine receipt");
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        first_window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &frame, window, cx)
                    })
                })
            })
            .expect("current shared receipt");
        for window in [&mut *first_window, &mut *second_window] {
            paint_presentation(window, &owner);
            assert!(window.update(|window, _| window.has_image_atlas_entry(frame.image())));
        }
        let next = session
            .update(first_window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("same source new generation")
            .await
            .expect("genuine new output");
        first_window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &next, window, cx)
                    })
                })
            })
            .expect("shared replacement");
        for window in [&mut *first_window, &mut *second_window] {
            assert!(!window.update(|window, _| window.has_image_atlas_entry(frame.image())));
            paint_presentation(window, &owner);
            assert!(window.update(|window, _| window.has_image_atlas_entry(next.image())));
        }
        drop(owner);
        drop(first_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_some());
        assert!(first_window.update(|window, _| window.has_image_atlas_entry(next.image())));
        drop(second_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(next.image())));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(next.image())));
        assert!(session.read_with(first_window, |session, _| session.preview_is_current(&next)));
        assert!(next.image().as_bytes(0).is_some());
    }
    fn encoded_artboard_presentation_document() -> (
        graphics_core::evaluate::CpuSvgArtboardDocument,
        Arc<graphics_core::resources::PreparedEncodedPngSource>,
        Arc<[u8]>,
    ) {
        use graphics_core::evaluate::{
            CpuSvgArtboardDocument, CpuSvgEncodedPngInput, CpuSvgEvaluationBudget,
        };
        use graphics_core::resources::{EncodedPngBudget, prepare_encoded_png_source};
        use graphite_glam::{DAffine2, DVec2};
        use graphite_graph_craft::document::NodeId;
        use sha2::{Digest, Sha256};
        let samples = [
            255, 0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 0, 128,
        ];
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, 2, 2);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("actual PNG header");
            writer
                .write_chunk(
                    png::chunk::tEXt,
                    b"workflow\0encoded Artboard preview fixture",
                )
                .expect("retained producer text");
            writer
                .write_image_data(&samples)
                .expect("asymmetric RGBA pixels");
            writer.finish().expect("complete PNG envelope");
        }
        let bytes: Arc<[u8]> = encoded.into();
        let source = Arc::new(
            prepare_encoded_png_source(
                ImageResourceIdentity {
                    asset_id: "test/input/encoded-artboard-preview.png".into(),
                    version_id: "caller-supplied-test-label".into(),
                    content_sha256: Sha256::digest(bytes.as_ref()).into(),
                },
                bytes.clone(),
                EncodedPngBudget {
                    max_encoded_bytes: 16 * 1024,
                    max_pixel_bytes: 4 * (4 + 2 * size_of::<graphite_std::Color>()),
                    max_metadata_bytes: 4096,
                },
            )
            .expect("genuine header/SHA seal without a production decode"),
        );
        let original = artboard_session_document();
        let mut parameters = original.parameters();
        parameters.location = DVec2::ZERO;
        parameters.dimensions = DVec2::new(24., 16.);
        parameters.background = graphite_std::Color::from_gamma_srgb_channels(0., 0., 0., 0.);
        let mut configuration = original.configuration();
        configuration.viewport.transform = DAffine2::IDENTITY;
        let placement =
            DAffine2::from_scale_angle_translation(DVec2::new(16., 12.), 0., DVec2::new(2., 2.));
        let input = CpuSvgEncodedPngInput::new(source.clone(), placement)
            .expect("explicit finite unit-square placement");
        let document = CpuSvgArtboardDocument::new_with_encoded_png(
            91,
            NodeId(7001),
            parameters,
            configuration,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes: 64 * 1024,
            },
            input,
        )
        .expect("genuine retained encoded Artboard source network");
        (document, source, bytes)
    }

    #[gpui::test]
    async fn at_r18_2_encoded_artboard_frame_preserves_original_source_metadata_and_visible_pixels(
        cx: &mut TestAppContext,
    ) {
        use graphite_glam::{DAffine2, DVec2};
        use graphite_graph_craft::document::{NodeId, value::RenderOutputType};
        use sha2::{Digest, Sha256};
        init_preview_view(cx);
        let (document, source, bytes) = encoded_artboard_presentation_document();
        let network = document.network().clone();
        let identity = source.identity().clone();
        let bytes_before = bytes.to_vec();
        let oracle = graphite_std::raster::Image::<graphite_std::Color>::from_encoded(&bytes)
            .expect("original decoder oracle outside the production request");
        assert_eq!((oracle.width, oracle.height), (2, 2));
        let oracle_before = oracle.to_flat_u8();
        assert_eq!(
            oracle_before.0,
            [
                255, 0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 0, 128
            ]
        );
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let preview = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("tracked genuine request")
            .await
            .expect("original graph and native conversion");
        let output_before = preview.output().output().clone();
        assert_eq!(preview.image().frame_count(), 1);
        assert_eq!(
            preview.image().size(0),
            size(DevicePixels(31), DevicePixels(17))
        );
        let pixels = preview
            .image()
            .as_bytes(0)
            .expect("single native straight-BGRA frame");
        assert_eq!(pixels.len(), 31 * 17 * 4);
        let pixel = |x: usize, y: usize| -> &[u8] {
            pixels
                .get((y * 31 + x) * 4..(y * 31 + x + 1) * 4)
                .expect("declared interior sample")
        };
        assert_eq!(pixel(0, 0), [0, 0, 0, 0]);
        let red = pixel(6, 5);
        assert!(
            red[2] > 180 && red[0] < 80 && red[1] < 80 && red[3] > 220,
            "red interior {red:?}"
        );
        let blue = pixel(14, 5);
        assert!(
            blue[0] > 180 && blue[1] < 80 && blue[2] < 80 && blue[3] > 220,
            "blue interior {blue:?}"
        );
        let green = pixel(6, 11);
        assert!(
            green[1] > 180 && green[0] < 80 && green[2] < 80 && green[3] > 220,
            "green interior {green:?}"
        );
        let yellow = pixel(14, 11);
        assert!(
            yellow[2] > 180
                && yellow[1] > 180
                && yellow[0] < 80
                && (100..=180).contains(&yellow[3]),
            "translucent yellow interior {yellow:?}"
        );
        let metadata = &preview.output().output().metadata;
        assert_eq!(metadata.backgrounds.len(), 1);
        assert_eq!(metadata.backgrounds[0].location, DVec2::ZERO);
        assert_eq!(metadata.backgrounds[0].dimensions, DVec2::new(24., 16.));
        assert_eq!(
            metadata.local_transforms.get(&NodeId(7001)),
            Some(&DAffine2::IDENTITY)
        );
        assert_eq!(
            metadata.clip_targets,
            std::collections::HashSet::from([NodeId(7001)])
        );
        assert_eq!(
            metadata
                .click_targets
                .get(&NodeId(7001))
                .expect("genuine source target")[0]
                .bounding_box(),
            Some([DVec2::ZERO, DVec2::new(24., 16.)])
        );
        let RenderOutputType::Svg { svg, image_data } = &preview.output().output().data else {
            panic!("actual CPU SVG output")
        };
        assert!(image_data.is_empty());
        assert_eq!(svg.matches("data:image/png;base64,").count(), 1);
        assert!(svg.contains("<image width=\"1\" height=\"1\""));
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &preview, window, cx)
                    })
                })
            })
            .expect("genuine foreground receipt publication");
        for _ in 0..3 {
            paint_presentation(window, &owner);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                    .is_none()
            );
            assert!(Arc::ptr_eq(
                &presented_image(&owner, window),
                preview.image()
            ));
            assert!(window.update(|window, _| window.has_image_atlas_entry(preview.image())));
            assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
        }
        session.read_with(window, |session, _| {
            assert_eq!(session.document().network(), &network);
            assert_eq!(
                session
                    .current_output()
                    .expect("current genuine output")
                    .output(),
                &output_before
            );
            let retained = session
                .document()
                .encoded_png_input()
                .expect("source retained");
            assert!(Arc::ptr_eq(retained.source(), &source));
        });
        assert_eq!(source.identity(), &identity);
        assert_eq!((source.width(), source.height()), (2, 2));
        assert_eq!(
            source.identity().content_sha256,
            <[u8; 32]>::from(Sha256::digest(bytes.as_ref()))
        );
        assert_eq!(bytes.as_ref(), bytes_before.as_slice());
        assert_eq!(oracle.to_flat_u8(), oracle_before);
        assert_eq!(preview.output().output(), &output_before);
    }

    #[gpui::test]
    async fn at_r18_2_encoded_artboard_edits_cap_cancel_and_recovery_keep_last_valid_source(
        cx: &mut TestAppContext,
    ) {
        use graphite_glam::{DAffine2, DVec2};
        init_preview_view(cx);
        let (document, source, bytes) = encoded_artboard_presentation_document();
        let placement = document
            .encoded_png_input()
            .expect("source placement")
            .placement();
        let bytes_before = bytes.to_vec();
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let first = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("first request")
            .await
            .expect("first encoded frame");
        let first_output = first.output().output().clone();
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &first, window, cx)
                    })
                })
            })
            .expect("first publication");
        paint_presentation(window, &owner);
        let mut parameters = session.read_with(window, |session, _| session.parameters());
        parameters.background = graphite_std::Color::from_gamma_srgb_channels(0., 0., 1., 0.5);
        let mut configuration = session.read_with(window, |session, _| session.configuration());
        session.update(window, |session, cx| {
            assert!(
                session
                    .apply(session.revision(), parameters, configuration, cx)
                    .expect("parameter edit")
            )
        });
        session.update(window, |session, cx| {
            owner.update(cx, |owner, cx| owner.refresh_artboard_status(session, cx))
        });
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), first.image()));
        assert!(!session.read_with(window, |session, _| session.preview_is_current(&first)));
        for maximum in [0, 31 * 17 * 4 - 1] {
            let failed = session
                .update(window, |session, cx| session.request_preview(maximum, cx))
                .expect("bounded request")
                .await;
            assert!(matches!(
                failed,
                Err(crate::session::ArtboardSessionError::Conversion(_))
            ));
            session.update(window, |session, cx| {
                owner.update(cx, |owner, cx| owner.refresh_artboard_status(session, cx))
            });
            paint_presentation(window, &owner);
            assert!(Arc::ptr_eq(&presented_image(&owner, window), first.image()));
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_some()
            );
            assert!(session.read_with(window, |session, _| session.diagnostic().is_some()));
        }
        let edited = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("exact cap recovery")
            .await
            .expect("current edited frame");
        assert_ne!(edited.image().as_bytes(0), first.image().as_bytes(0));
        assert_eq!(
            &edited.image().as_bytes(0).expect("edited pixels")[..4],
            [254, 0, 0, 128]
        );
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &edited, window, cx)
                    })
                })
            })
            .expect("recovery publication");
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(first.image())));
        configuration.viewport.transform = DAffine2::from_translation(DVec2::new(3., 0.));
        session.update(window, |session, cx| {
            assert!(
                session
                    .apply(session.revision(), parameters, configuration, cx)
                    .expect("context successor")
            )
        });
        let shifted = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("shifted context request")
            .await
            .expect("shifted original renderer frame");
        assert_ne!(shifted.image().as_bytes(0), edited.image().as_bytes(0));
        assert_eq!(
            window.update(|window, cx| session
                .update(cx, |session, cx| owner.update(cx, |owner, cx| owner
                    .accept_artboard_preview(session, &edited, window, cx)))),
            Err(CpuPreviewError::ResultNotFresh)
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            edited.image()
        ));
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &shifted, window, cx)
                    })
                })
            })
            .expect("context recovery publication");
        let cancelled = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("pending request");
        session.update(window, |session, cx| assert!(session.cancel(cx)));
        assert!(cancelled.await.is_err());
        session.update(window, |session, cx| {
            owner.update(cx, |owner, cx| owner.refresh_artboard_status(session, cx))
        });
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(Arc::ptr_eq(
            &presented_image(&owner, window),
            shifted.image()
        ));
        assert!(!session.read_with(window, |session, _| session.preview_is_current(&shifted)));
        let recovered = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("cancel recovery request")
            .await
            .expect("genuine cancel recovery");
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &recovered, window, cx)
                    })
                })
            })
            .expect("cancel recovery publication");
        paint_presentation(window, &owner);
        assert!(owner.read_with(window, |owner, _| owner.is_fresh()));
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert_eq!(recovered.image().as_bytes(0), shifted.image().as_bytes(0));
        session.read_with(window, |session, _| {
            let retained = session
                .document()
                .encoded_png_input()
                .expect("retained encoded input after all edits");
            assert!(Arc::ptr_eq(retained.source(), &source));
            assert_eq!(
                retained.placement().to_cols_array().map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(session.configuration(), configuration);
            assert_eq!(session.parameters(), parameters);
            assert!(session.preview_is_current(&recovered));
        });
        assert_eq!(first.output().output(), &first_output);
        assert_eq!(bytes.as_ref(), bytes_before.as_slice());
    }

    #[gpui::test]
    async fn at_r18_2_encoded_artboard_shared_frame_replacement_and_release_retire_both_atlases(
        cx: &mut TestAppContext,
    ) {
        init_preview_view(cx);
        let (document, source, bytes) = encoded_artboard_presentation_document();
        let bytes_before = bytes.to_vec();
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view = owner.clone();
        let second_view = owner.clone();
        let weak_owner = owner.downgrade();
        let frame = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("encoded request")
            .await
            .expect("genuine encoded frame");
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        first_window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &frame, window, cx)
                    })
                })
            })
            .expect("single shared publication");
        for window in [&mut *first_window, &mut *second_window] {
            paint_presentation(window, &owner);
            assert!(Arc::ptr_eq(&presented_image(&owner, window), frame.image()));
            assert!(window.update(|window, _| window.has_image_atlas_entry(frame.image())));
            assert!(session.read_with(window, |session, _| session.preview_is_current(&frame)));
        }
        let repeated = session.read_with(first_window, |session, _| {
            session.preview().expect("same retained request").clone()
        });
        assert_eq!(repeated.generation(), frame.generation());
        assert!(Arc::ptr_eq(repeated.image(), frame.image()));
        let next = session
            .update(first_window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("explicit successor")
            .await
            .expect("same source successor frame");
        assert_ne!(next.generation(), frame.generation());
        assert_eq!(next.image().as_bytes(0), frame.image().as_bytes(0));
        assert!(!session.read_with(first_window, |session, _| {
            session.preview_is_current(&frame)
        }));
        first_window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &next, window, cx)
                    })
                })
            })
            .expect("explicit shared replacement");
        for window in [&mut *first_window, &mut *second_window] {
            assert!(!window.update(|window, _| window.has_image_atlas_entry(frame.image())));
            paint_presentation(window, &owner);
            assert!(Arc::ptr_eq(&presented_image(&owner, window), next.image()));
            assert!(window.update(|window, _| window.has_image_atlas_entry(next.image())));
        }
        drop(owner);
        drop(first_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_some());
        drop(second_view);
        first_window.cx.update(|_| {});
        assert!(weak_owner.upgrade().is_none());
        assert!(!first_window.update(|window, _| window.has_image_atlas_entry(next.image())));
        assert!(!second_window.update(|window, _| window.has_image_atlas_entry(next.image())));
        assert!(session.read_with(first_window, |session, _| session.preview_is_current(&next)));
        session.read_with(first_window, |session, _| {
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source retained independently of view")
                    .source(),
                &source
            ))
        });
        assert_eq!(bytes.as_ref(), bytes_before.as_slice());
        assert_eq!(next.image().as_bytes(0), frame.image().as_bytes(0));
        let weak_session = session.downgrade();
        let weak_frame = Arc::downgrade(next.image());
        drop(frame);
        drop(repeated);
        drop(next);
        assert!(weak_frame.upgrade().is_some());
        drop(session);
        first_window.cx.update(|_| {});
        assert!(weak_session.upgrade().is_none());
        assert!(weak_frame.upgrade().is_none());
    }
    #[gpui::test]
    async fn at_r22_1_encoded_artboard_png_matches_native_frame_and_retains_source_after_edit(
        cx: &mut TestAppContext,
    ) {
        use graphite_graph_craft::document::NodeId;
        use sha2::{Digest, Sha256};
        init_preview_view(cx);
        let (mut document, source, bytes) = encoded_artboard_presentation_document();
        let source_identity = source.identity().clone();
        let placement = document
            .encoded_png_input()
            .expect("explicit encoded placement")
            .placement();
        let original_network = document.network().clone();
        let original_parameters = document.parameters();
        let original_configuration = document.configuration();
        let detached = document.submit();
        let bytes_before = bytes.to_vec();
        assert!(
            bytes
                .windows(b"workflow\0encoded Artboard preview fixture".len())
                .any(|window| window == b"workflow\0encoded Artboard preview fixture")
        );
        let session = cx.new(|_| crate::session::GraphiteArtboardSession::new(document));
        let first = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("genuine encoded Artboard preview request")
            .await
            .expect("original decoded scene and native frame");
        let first_output = first.output().output().clone();
        let first_frame = first
            .image()
            .as_bytes(0)
            .expect("captured native frame")
            .to_vec();
        let first_png = session
            .update(cx, |session, cx| {
                session.prepare_png(first.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("genuine current preview bake")
            .await
            .expect("baked PNG from retained encoded Artboard");
        let inspect = |prepared: &crate::session::PreparedArtboardPngAsset,
                       preview: &crate::session::PreparedArtboardPreview| {
            let pixels = preview
                .image()
                .as_bytes(0)
                .expect("one captured straight-BGRA frame");
            assert_eq!(preview.image().frame_count(), 1);
            assert_eq!(
                preview.image().size(0),
                size(DevicePixels(31), DevicePixels(17))
            );
            assert_eq!(pixels.len(), 31 * 17 * 4);
            let rgba: Vec<u8> = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[2], pixel[1], pixel[0], pixel[3]])
                .collect();
            assert!(rgba.as_chunks::<4>().0.iter().any(|pixel| pixel[3] > 0));
            let png = prepared.png();
            assert_eq!((png.width(), png.height()), (31, 17));
            let decoded =
                image::load_from_memory_with_format(png.encoded_bytes(), image::ImageFormat::Png)
                    .expect("actual baked PNG decoder")
                    .into_rgba8();
            assert_eq!(decoded.dimensions(), (31, 17));
            assert_eq!(decoded.as_raw(), &rgba);
            let pixel_digest: [u8; 32] = Sha256::digest(&rgba).into();
            let encoded_digest: [u8; 32] = Sha256::digest(png.encoded_bytes()).into();
            assert_eq!(prepared.pixels_sha256(), pixel_digest);
            assert_eq!(png.content_sha256(), encoded_digest);
            assert_ne!(pixel_digest, encoded_digest);
            assert_eq!(png.sample_format(), CpuSampleFormat::Rgba8);
            assert_eq!(png.color_space(), ImageColorSpace::Srgb);
            assert_eq!(png.alpha(), AlphaAssociation::Straight);
            assert_eq!(png.content_type(), "image/png");
            let direct = graphics_core::export::prepare_rgba8_png(31, 17, rgba, 64 * 1024)
                .expect("same shared byte encoder oracle");
            assert_eq!(png.encoded_bytes(), direct.png().encoded_bytes());
            assert!(Arc::ptr_eq(
                &prepared.source_output().output_lease(),
                &preview.output().output_lease()
            ));
            assert_eq!(prepared.source_output().output(), preview.output().output());
            assert_eq!(
                prepared.source_output().output().metadata,
                preview.output().output().metadata
            );
            assert_eq!(
                prepared.source_output().configuration(),
                preview.output().configuration()
            );
            assert_eq!(prepared.source_output().document_id_encoding(), 91);
            assert_eq!(prepared.source_output().node_id(), NodeId(7001));
            assert_eq!(
                prepared.source_output().revision(),
                preview.output().revision()
            );
        };
        inspect(&first_png, &first);
        let first_png_bytes = first_png.png().encoded_bytes().to_vec();
        session.read_with(cx, |session, _| {
            assert!(session.preview_is_current(&first));
            assert!(session.png_is_current(&first_png));
            assert_eq!(session.document().network(), &original_network);
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("encoded source retained")
                    .source(),
                &source
            ));
        });
        let mut parameters = session.read_with(cx, |session, _| session.parameters());
        parameters.background = graphite_std::Color::from_gamma_srgb_channels(0., 0., 1., 0.5);
        let configuration = session.read_with(cx, |session, _| session.configuration());
        session.update(cx, |session, cx| {
            assert!(
                session
                    .apply(session.revision(), parameters, configuration, cx)
                    .expect("original Artboard background edit")
            )
        });
        assert!(!session.read_with(cx, |session, _| session.png_is_current(&first_png)));
        assert!(!session.read_with(cx, |session, _| session.preview_is_current(&first)));
        assert!(matches!(
            session.update(cx, |session, cx| session.prepare_png(
                first.clone(),
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(crate::session::ArtboardSessionError::Superseded)
        ));
        let edited_network =
            session.read_with(cx, |session, _| session.document().network().clone());
        assert_ne!(edited_network, original_network);
        let edited = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("explicit edited preview request")
            .await
            .expect("original edited encoded Artboard frame");
        assert_ne!(edited.image().as_bytes(0), Some(first_frame.as_slice()));
        let edited_png = session
            .update(cx, |session, cx| {
                session.prepare_png(edited.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("fresh edited PNG request")
            .await
            .expect("new derived prepared bytes");
        inspect(&edited_png, &edited);
        assert_ne!(edited_png.png().encoded_bytes(), first_png_bytes.as_slice());
        assert_ne!(
            edited_png.png().content_sha256(),
            first_png.png().content_sha256()
        );
        assert_ne!(edited_png.pixels_sha256(), first_png.pixels_sha256());
        assert_eq!(
            &edited.image().as_bytes(0).expect("edited frame")[..4],
            [254, 0, 0, 128]
        );
        session.read_with(cx, |session, _| {
            assert!(session.preview_is_current(&edited));
            assert!(session.png_is_current(&edited_png));
            assert!(!session.png_is_current(&first_png));
            assert_eq!(session.document().network(), &edited_network);
            assert_eq!(session.parameters(), parameters);
            assert_eq!(session.configuration(), configuration);
            let retained = session
                .document()
                .encoded_png_input()
                .expect("encoded input retained through editing and PNG preparation");
            assert!(Arc::ptr_eq(retained.source(), &source));
            assert_eq!(
                retained.placement().to_cols_array().map(f64::to_bits),
                placement.to_cols_array().map(f64::to_bits)
            );
        });
        assert_eq!(detached.network(), &original_network);
        assert_eq!(source.identity(), &source_identity);
        assert_eq!((source.width(), source.height()), (2, 2));
        assert_eq!(
            source.identity().content_sha256,
            <[u8; 32]>::from(Sha256::digest(bytes.as_ref()))
        );
        assert_eq!(bytes.as_ref(), bytes_before.as_slice());
        assert_eq!(first.image().as_bytes(0), Some(first_frame.as_slice()));
        assert_eq!(first_png.png().encoded_bytes(), first_png_bytes.as_slice());
        assert_eq!(first_png.source_output().output(), &first_output);
        assert_eq!(first.output().output(), &first_output);
        let inspect_checkpoint =
            |prepared: &crate::session::PreparedArtboardPngAsset,
             expected_parameters: graphics_core::evaluate::CpuSvgArtboardParameters| {
                let accepted = prepared.source_output();
                assert_eq!(accepted.parameters(), expected_parameters);
                assert_eq!(accepted.configuration(), original_configuration);
                let retained = accepted
                    .encoded_png_input()
                    .expect("source owned by the exact accepted checkpoint");
                assert!(Arc::ptr_eq(retained.source(), &source));
                assert_eq!(retained.source().identity(), &source_identity);
                assert_eq!(
                    retained.placement().to_cols_array().map(f64::to_bits),
                    placement.to_cols_array().map(f64::to_bits)
                );
            };
        inspect_checkpoint(&first_png, original_parameters);
        inspect_checkpoint(&edited_png, parameters);
        let weak_session = session.downgrade();
        drop(session);
        cx.update(|_| {});
        cx.run_until_parked();
        assert!(weak_session.upgrade().is_none());
        inspect_checkpoint(&first_png, original_parameters);
        inspect_checkpoint(&edited_png, parameters);
        inspect(&first_png, &first);
        inspect(&edited_png, &edited);
        assert_eq!(first_png.source_output().output(), &first_output);
        assert_eq!(first_png.png().encoded_bytes(), first_png_bytes.as_slice());
        assert_eq!(detached.network(), &original_network);
        assert_eq!(bytes.as_ref(), bytes_before.as_slice());
    }
    fn host_bound_artboard_request(
        png: &crate::resources::tests::IndexedRequestPng,
    ) -> crate::interop::GraphiteAssetArtboardRequest {
        use graphite_glam::{DAffine2, DVec2};
        let mut request = crate::interop::artboard_binding_controls::request(png);
        request.parameters.background =
            graphite_std::Color::from_gamma_srgb_channels(0., 0., 0., 0.);
        request.placement =
            DAffine2::from_scale_angle_translation(DVec2::new(18., 8.), 0., DVec2::new(2., 2.));
        request
    }

    #[gpui::test]
    async fn at_r28_1_host_artboard_revocation_hides_both_windows_and_denies_png_without_losing_document(
        cx: &mut TestAppContext,
    ) {
        use crate::interop::{GraphiteAssetBinding, GraphiteAssetBindingError};
        use crate::resources::tests::{host_reader_fixture, indexed_request_png};
        use crate::session::ArtboardSessionError;
        init_preview_view(cx);
        let fixture =
            host_reader_fixture().expect("real local host service and sealed Input grant");
        let png = indexed_request_png(&fixture, 4).expect("actual indexed producer RGBA PNG");
        let record_before = fixture
            .assets
            .lock()
            .expect("real asset index")
            .record(&png.asset_identity)
            .expect("indexed immutable producer record");
        let binding = cx
            .update(|cx| {
                GraphiteAssetBinding::new(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("current real service/profile binding");
        let session = binding
            .update(cx, |binding, cx| {
                binding.open_artboard(host_bound_artboard_request(&png), cx)
            })
            .expect("genuine authorized asynchronous acquisition")
            .await
            .expect("one verified Input read and retained source network");
        let source = session.read_with(cx, |session, _| {
            session
                .document()
                .encoded_png_input()
                .expect("retained verified source")
                .source()
                .clone()
        });
        assert_eq!(source.identity(), &png.image_identity);
        assert_eq!((source.width(), source.height()), (3, 1));
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        let parameters_before = session.read_with(cx, |session, _| session.parameters());
        let configuration_before = session.read_with(cx, |session, _| session.configuration());
        let preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("current bound request")
            .await
            .expect("genuine original graph and native frame");
        let output_before = preview.output().output().clone();
        let pixels_before = preview
            .image()
            .as_bytes(0)
            .expect("native BGRA frame")
            .to_vec();
        assert_eq!(
            preview.image().size(0),
            size(DevicePixels(31), DevicePixels(17))
        );
        assert_eq!(pixels_before.len(), 31 * 17 * 4);
        let red = pixels_before
            .get((5 * 31 + 4) * 4..(5 * 31 + 5) * 4)
            .expect("interior opaque source texel");
        assert!(
            red[2] > 180 && red[0] < 80 && red[1] < 80 && red[3] > 220,
            "original producer red interior {red:?}"
        );
        let prepared = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("fresh PNG start")
            .await
            .expect("genuine current PNG");
        let encoded_before = prepared.png().encoded_bytes().to_vec();
        assert!(session.read_with(cx, |session, _| session.png_is_current(&prepared)));
        let owner = cx.new(CpuPreviewPresentation::new);
        let first_view = owner.clone();
        let second_view = owner.clone();
        let mut second_context = cx.clone();
        let first_window = cx.add_empty_window();
        let second_window = second_context.add_empty_window();
        first_window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &preview, window, cx)
                    })
                })
            })
            .expect("current bound frame publication");
        for window in [&mut *first_window, &mut *second_window] {
            paint_presentation(window, &owner);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(Arc::ptr_eq(
                &presented_image(&owner, window),
                preview.image()
            ));
            assert!(window.update(|window, _| window.has_image_atlas_entry(preview.image())));
        }
        binding
            .update(first_window, |binding, cx| binding.invalidate(cx))
            .expect("invalidate binding before revoking all weak sessions");
        assert!(!binding.read_with(first_window, |binding, _| binding.is_active()));
        session.read_with(first_window, |session, _| {
            assert!(session.resource_is_revoked());
            assert!(!session.preview_is_current(&preview));
            assert!(!session.png_is_current(&prepared));
            assert!(session.preview().is_none());
            assert!(session.current_output().is_none());
            assert!(!session.is_pending());
            assert!(matches!(
                session.diagnostic(),
                Some(ArtboardSessionError::ResourceRevoked)
            ));
            assert_eq!(session.document().network(), &network_before);
            assert_eq!(session.parameters(), parameters_before);
            assert_eq!(session.configuration(), configuration_before);
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("editable source retained after permission loss")
                    .source(),
                &source
            ));
        });
        first_window.update(|window, cx| {
            session.update(cx, |session, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_artboard_resource_status(session, window, cx)
                })
            })
        });
        for window in [&mut *first_window, &mut *second_window] {
            assert!(!window.update(|window, _| window.has_image_atlas_entry(preview.image())));
            paint_presentation(window, &owner);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                    .is_none()
            );
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                    .is_some()
            );
            assert!(owner.read_with(window, |owner, _| owner.image().is_none()));
            assert!(matches!(
                owner.read_with(window, |owner, _| owner.diagnostic().cloned()),
                Some(CpuPreviewDiagnostic::Preparation(
                    CpuPreviewError::ResourceRevoked
                ))
            ));
        }
        assert_eq!(
            first_window.update(|window, cx| session
                .update(cx, |session, cx| owner.update(cx, |owner, cx| owner
                    .accept_artboard_preview(session, &preview, window, cx)))),
            Err(CpuPreviewError::ResourceRevoked)
        );
        assert!(matches!(
            session.update(first_window, |session, cx| session.prepare_png(
                preview.clone(),
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(ArtboardSessionError::ResourceRevoked)
        ));
        let mut edited_parameters = parameters_before;
        edited_parameters.dimensions.x = 22.;
        session.update(first_window, |session, cx| {
            assert!(
                session
                    .apply(
                        session.revision(),
                        edited_parameters,
                        configuration_before,
                        cx
                    )
                    .expect("metadata stays editable after permission loss")
            )
        });
        assert!(matches!(
            session.update(first_window, |session, cx| session
                .request_preview(31 * 17 * 4, cx)),
            Err(ArtboardSessionError::ResourceRevoked)
        ));
        assert!(matches!(
            session.update(first_window, |session, cx| session.prepare_png(
                preview.clone(),
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(ArtboardSessionError::ResourceRevoked)
        ));
        session.read_with(first_window, |session, _| {
            assert!(session.resource_is_revoked());
            assert_eq!(session.parameters(), edited_parameters);
            assert!(matches!(
                session.diagnostic(),
                Some(ArtboardSessionError::ResourceRevoked)
            ));
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("source retained but not authorized for evaluation")
                    .source(),
                &source
            ));
        });
        assert!(matches!(
            binding.update(first_window, |binding, cx| binding
                .open_artboard(host_bound_artboard_request(&png), cx)),
            Err(GraphiteAssetBindingError::Inactive)
        ));
        assert_eq!(preview.output().output(), &output_before);
        assert_eq!(preview.image().as_bytes(0), Some(pixels_before.as_slice()));
        assert_eq!(prepared.png().encoded_bytes(), encoded_before.as_slice());
        assert_eq!(prepared.source_output().parameters(), parameters_before);
        assert_eq!(prepared.source_output().output(), &output_before);
        assert!(Arc::ptr_eq(
            prepared
                .source_output()
                .encoded_png_input()
                .expect("escaped historical checkpoint retains source")
                .source(),
            &source
        ));
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("index remains intact")
                .record(&png.asset_identity),
            Some(record_before)
        );
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("same real service")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("canonical verified input remains unchanged")
                .as_slice(),
            png.encoded.as_slice()
        );
        let recovered_binding = first_window
            .update(|_, cx| {
                GraphiteAssetBinding::new(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("explicit fresh host binding");
        let recovered_session = recovered_binding
            .update(first_window, |binding, cx| {
                binding.open_artboard(host_bound_artboard_request(&png), cx)
            })
            .expect("explicit new binding request")
            .await
            .expect("genuine recovery acquisition");
        let recovered = recovered_session
            .update(first_window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("fresh recovery request")
            .await
            .expect("fresh recovery frame");
        first_window
            .update(|window, cx| {
                recovered_session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &recovered, window, cx)
                    })
                })
            })
            .expect("only fresh binding can publish recovery");
        for window in [&mut *first_window, &mut *second_window] {
            paint_presentation(window, &owner);
            assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
            assert!(
                window
                    .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                    .is_none()
            );
            assert!(Arc::ptr_eq(
                &presented_image(&owner, window),
                recovered.image()
            ));
        }
        assert!(session.read_with(first_window, |session, _| session.resource_is_revoked()));
        assert_eq!(
            recovered.image().as_bytes(0),
            Some(pixels_before.as_slice())
        );
        drop(first_view);
        drop(second_view);
    }

    #[gpui::test]
    async fn at_r28_1_host_artboard_cancel_keeps_last_valid_and_recovers_without_reviving_revoked_leases(
        cx: &mut TestAppContext,
    ) {
        use crate::interop::GraphiteAssetBinding;
        use crate::resources::tests::{host_reader_fixture, indexed_request_png};
        use crate::session::ArtboardSessionError;
        init_preview_view(cx);
        let fixture = host_reader_fixture().expect("real indexed service and grant fixture");
        let png = indexed_request_png(&fixture, 3).expect("actual original RGB producer input");
        let binding = cx
            .update(|cx| {
                GraphiteAssetBinding::new(
                    fixture.assets.clone(),
                    fixture.input_authorization.clone(),
                    cx,
                )
            })
            .expect("active service/profile binding");
        let session = binding
            .update(cx, |binding, cx| {
                binding.open_artboard(host_bound_artboard_request(&png), cx)
            })
            .expect("authorized acquisition start")
            .await
            .expect("genuine retained encoded input");
        let source = session.read_with(cx, |session, _| {
            session
                .document()
                .encoded_png_input()
                .expect("verified source")
                .source()
                .clone()
        });
        let network = session.read_with(cx, |session, _| session.document().network().clone());
        let owner = cx.new(CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let first = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("bound preview")
            .await
            .expect("original renderer frame");
        let output_before = first.output().output().clone();
        assert!(
            first
                .image()
                .as_bytes(0)
                .expect("visible frame")
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[3] > 0)
        );
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &first, window, cx)
                    })
                })
            })
            .expect("fresh frame acceptance");
        let pending = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("tracked pending successor");
        assert!(session.read_with(window, |session, _| session.is_pending()));
        session.update(window, |session, cx| assert!(session.cancel(cx)));
        assert!(pending.await.is_err());
        window.update(|window, cx| {
            session.update(cx, |session, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_artboard_resource_status(session, window, cx)
                })
            })
        });
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_some()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(Arc::ptr_eq(&presented_image(&owner, window), first.image()));
        assert!(window.update(|window, _| window.has_image_atlas_entry(first.image())));
        assert!(matches!(
            session.update(window, |session, cx| session.prepare_png(
                first.clone(),
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(ArtboardSessionError::Superseded)
        ));
        session.read_with(window, |session, _| {
            assert!(!session.resource_is_revoked());
            assert!(!session.preview_is_current(&first));
            assert_eq!(session.document().network(), &network);
            assert!(Arc::ptr_eq(
                session
                    .document()
                    .encoded_png_input()
                    .expect("cancelled source retained")
                    .source(),
                &source
            ));
        });
        assert!(binding.read_with(window, |binding, _| binding.is_active()));
        let recovered = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("same active binding explicit retry")
            .await
            .expect("genuine cancellation recovery");
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    owner.update(cx, |owner, cx| {
                        owner.accept_artboard_preview(session, &recovered, window, cx)
                    })
                })
            })
            .expect("fresh recovered frame");
        paint_presentation(window, &owner);
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_none()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(first.image())));
        assert_eq!(recovered.image().as_bytes(0), first.image().as_bytes(0));
        let png_result = session
            .update(window, |session, cx| {
                session.prepare_png(recovered.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("recovered current PNG request")
            .await
            .expect("current PNG allowed without new grant fabrication");
        assert!(session.read_with(window, |session, _| session.png_is_current(&png_result)));
        binding
            .update(window, |binding, cx| binding.invalidate(cx))
            .expect("later genuine terminal invalidation");
        let denied = session.update(window, |session, cx| {
            session.request_preview(31 * 17 * 4, cx)
        });
        assert!(matches!(denied, Err(ArtboardSessionError::ResourceRevoked)));
        window.update(|window, cx| {
            session.update(cx, |session, cx| {
                owner.update(cx, |owner, cx| {
                    owner.refresh_artboard_resource_status(session, window, cx)
                })
            })
        });
        paint_presentation(window, &owner);
        assert!(window.debug_bounds("GRAPHITE-CPU-PREVIEW-IMAGE").is_none());
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-LAST-VALID")
                .is_none()
        );
        assert!(
            window
                .debug_bounds("GRAPHITE-CPU-PREVIEW-DIAGNOSTIC")
                .is_some()
        );
        assert!(!window.update(|window, _| window.has_image_atlas_entry(recovered.image())));
        assert!(session.read_with(window, |session, _| session.resource_is_revoked()));
        assert!(!session.read_with(window, |session, _| session.png_is_current(&png_result)));
        assert_eq!(first.output().output(), &output_before);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("canonical service")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("source bytes unchanged")
                .as_slice(),
            png.encoded.as_slice()
        );
    }
}

#[cfg(test)]
mod mix_pixels_preview_tests {
    use super::*;
    use gpui::{DevicePixels, size};
    use graphics_core::resources::{
        AlphaAssociation, CpuBlendMode, CpuImageBudget, CpuImageDescriptor, CpuImageEdit,
        CpuMixBudget, CpuMixError, CpuMixParameters, CpuSampleFormat, ImageColorSpace,
        ImageResourceIdentity, admit_cpu_image, evaluate_cpu_image_edit, evaluate_cpu_image_mix,
    };
    use sha2::{Digest, Sha256};

    const IDENTITY: [f64; 6] = [1., 0., 0., 1., 0., 0.];
    const OVER: [u8; 16] = [
        255, 0, 0, 255, 0, 0, 255, 128, 64, 128, 192, 64, 83, 47, 19, 0,
    ];
    const UNDER: [u8; 16] = [
        0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 128, 17, 201, 83, 0,
    ];

    fn parent(asset: &str, bytes: &[u8], transform: [f64; 6]) -> AdmittedCpuImage {
        admit_cpu_image(
            ImageResourceIdentity {
                asset_id: asset.into(),
                version_id: "test-supplied-version".into(),
                content_sha256: Sha256::digest(bytes).into(),
            },
            CpuImageDescriptor {
                width: 2,
                height: 2,
                row_stride: 8,
                sample_format: CpuSampleFormat::Rgba8,
                color_space: ImageColorSpace::Srgb,
                alpha: AlphaAssociation::Straight,
                transform,
            },
            Arc::from(bytes),
            CpuImageBudget {
                max_input_bytes: bytes.len(),
                max_decoded_bytes: 4096,
            },
        )
        .expect("original admitted parent pixels")
    }

    fn budget() -> CpuMixBudget {
        CpuMixBudget {
            max_parent_bytes: 4096,
            max_result_bytes: 4096,
        }
    }

    #[test]
    fn at_gra_node_143_3_mix_pixels_preview_matches_original_conversion_without_mutating_parents() {
        let over = parent("test/over", &OVER, IDENTITY);
        let under = parent("test/under", &UNDER, IDENTITY);
        let before = (
            over.image().clone(),
            under.image().clone(),
            over.identity().clone(),
            under.identity().clone(),
            *over.descriptor(),
            *under.descriptor(),
        );
        for opacity in [0., 37.5, 100.] {
            let result = evaluate_cpu_image_mix(
                &over,
                &under,
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity,
                },
                budget(),
            )
            .expect("pinned original Mix result");
            let original_image = result.image().clone();
            let attributes = result.attributes().clone();
            let digest = result.pixels_sha256();
            let float_bits: Vec<_> = result
                .image()
                .data
                .iter()
                .map(|color| [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits))
                .collect();
            let (mut original, width, height) = result.image().to_flat_u8();
            for pixel in original.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
            let first =
                prepare_cpu_pixels_preview(result.pixels(), 16).expect("shared Mix pixel preview");
            let repeated =
                prepare_cpu_pixels_preview(result.pixels(), 16).expect("repeat source conversion");
            assert_eq!((width, height), (2, 2));
            assert_eq!(first.frame_count(), 1);
            assert_eq!(first.size(0), size(DevicePixels(2), DevicePixels(2)));
            assert_eq!(first.as_bytes(0), Some(original.as_slice()));
            assert_eq!(first.as_bytes(1), None);
            assert_eq!(repeated.as_bytes(0), first.as_bytes(0));
            assert_eq!(
                first.as_bytes(0).expect("frame").get(12..16),
                Some(&[0, 0, 0, 0][..])
            );
            assert_eq!(result.parents()[0].identity(), over.identity());
            assert_eq!(result.parents()[1].identity(), under.identity());
            assert_eq!(result.image(), &original_image);
            assert_eq!(result.attributes(), &attributes);
            assert_eq!(result.pixels_sha256(), digest);
            assert_eq!(
                result
                    .image()
                    .data
                    .iter()
                    .map(|color| [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits))
                    .collect::<Vec<_>>(),
                float_bits
            );
        }
        assert_eq!(over.image(), &before.0);
        assert_eq!(under.image(), &before.1);
        assert_eq!(over.identity(), &before.2);
        assert_eq!(under.identity(), &before.3);
        assert_eq!(over.descriptor(), &before.4);
        assert_eq!(under.descriptor(), &before.5);
    }

    #[test]
    fn at_gra_node_143_3_mix_pixels_preview_exact_cap_and_failed_caps_preserve_float_result() {
        let over = parent("test/cap-over", &OVER, IDENTITY);
        let under = parent("test/cap-under", &UNDER, IDENTITY);
        let result = evaluate_cpu_image_mix(
            &over,
            &under,
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 50.,
            },
            budget(),
        )
        .expect("original Mix");
        let before = (
            result.image().clone(),
            result.attributes().clone(),
            result.pixels_sha256(),
            *result.descriptor(),
        );
        let exact = prepare_cpu_pixels_preview(result.pixels(), 16)
            .expect("exact additional preview-buffer cap");
        for cap in [0, 15] {
            assert_eq!(
                prepare_cpu_pixels_preview(result.pixels(), cap).err(),
                Some(CpuPreviewError::PreviewBudgetExceeded)
            );
            assert_eq!(result.image(), &before.0);
            assert_eq!(result.attributes(), &before.1);
            assert_eq!(result.pixels_sha256(), before.2);
            assert_eq!(result.descriptor(), &before.3);
        }
        let recovered = prepare_cpu_pixels_preview(result.pixels(), 16)
            .expect("recovery after rejected preparation");
        assert_eq!(recovered.as_bytes(0), exact.as_bytes(0));
        assert_eq!(over.identity(), result.parents()[0].identity());
        assert_eq!(under.identity(), result.parents()[1].identity());
    }

    #[test]
    fn at_gra_node_143_3_mix_pixels_preview_accepts_signed_zero_identity_without_rewriting_metadata()
     {
        let over_transform = [1., -0., 0., 1., -0., 0.];
        let under_transform = [1., 0., -0., 1., 0., -0.];
        let over = parent("test/signed-over", &OVER, over_transform);
        let under = parent("test/signed-under", &UNDER, under_transform);
        let result = evaluate_cpu_image_mix(
            &over,
            &under,
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 100.,
            },
            budget(),
        )
        .expect("numeric identity with exact source transform bits");
        let original_transform = result.descriptor().transform.map(f64::to_bits);
        let attributes = result.attributes().clone();
        assert_eq!(original_transform, over_transform.map(f64::to_bits));
        assert_eq!(
            result.parents()[0].descriptor().transform.map(f64::to_bits),
            over_transform.map(f64::to_bits)
        );
        assert_eq!(
            result.parents()[1].descriptor().transform.map(f64::to_bits),
            under_transform.map(f64::to_bits)
        );
        assert!(!result.requires_scene_render());
        let preview = prepare_cpu_pixels_preview(result.pixels(), 16)
            .expect("identity pixel-only preparation");
        assert_eq!(preview.size(0), size(DevicePixels(2), DevicePixels(2)));
        assert_eq!(
            result.descriptor().transform.map(f64::to_bits),
            original_transform
        );
        assert_eq!(result.attributes(), &attributes);
        assert_eq!(
            over.descriptor().transform.map(f64::to_bits),
            over_transform.map(f64::to_bits)
        );
        assert_eq!(
            under.descriptor().transform.map(f64::to_bits),
            under_transform.map(f64::to_bits)
        );
    }

    #[test]
    fn at_r08_21_2_shared_pixels_preview_rejects_scene_results_and_mix_transformed_parents() {
        let over = parent("test/scene-over", &OVER, [1., 0., 0., 1., 12., 0.]);
        let under = parent("test/identity-under", &UNDER, IDENTITY);
        let original = over.image().clone();
        assert_eq!(
            evaluate_cpu_image_mix(
                &over,
                &under,
                CpuMixParameters {
                    blend_mode: CpuBlendMode::Normal,
                    opacity: 100.
                },
                budget()
            )
            .expect_err("checked Mix cannot invent scene composition"),
            CpuMixError::SceneRenderRequired
        );
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            let result = evaluate_cpu_image_edit(&over, operation, 4096)
                .expect("original unary retains source scene attributes");
            let image = result.image().clone();
            let attributes = result.attributes().clone();
            assert!(result.requires_scene_render());
            for cap in [0, 16] {
                assert_eq!(
                    prepare_cpu_pixels_preview(result.pixels(), cap).err(),
                    Some(CpuPreviewError::SceneRenderRequired)
                );
                assert_eq!(
                    prepare_cpu_result_preview(&result, cap).err(),
                    Some(CpuPreviewError::SceneRenderRequired)
                );
            }
            assert_eq!(result.image(), &image);
            assert_eq!(result.attributes(), &attributes);
        }
        assert_eq!(over.image(), &original);
        assert_eq!(over.descriptor().transform, [1., 0., 0., 1., 12., 0.]);
        let identity_over = parent("test/identity-over", &OVER, IDENTITY);
        let valid = evaluate_cpu_image_mix(
            &identity_over,
            &under,
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 100.,
            },
            budget(),
        )
        .expect("explicit identity input");
        assert!(prepare_cpu_pixels_preview(valid.pixels(), 16).is_ok());
    }
}
