use crate::{NativeImageImportBinding, NativeImageOutputReader, native_asset_services};
use gpui::{
    AnyElement, App, AppContext as _, Context, FocusHandle, Image, ImageFormat, ImageSource,
    IntoElement, KeyDownEvent, ObjectFit, RenderOnce, Role, SharedString, StyledImage,
    Subscription, Task, Window, img,
};
use sim_media::{PngLimits, decode_png};
use sim_runtime::{
    AttemptId, AttemptPresentation, AttemptState, ExecutionDataSource, ExecutionFailure,
    ExecutionOutput, ExecutionOutputAvailability, ExecutionPreview, ExecutionSnapshot,
    ExecutionSnapshotStatus, OperationEligibility, OutputMediaKind,
};
use sim_types::CancellationToken;
use std::{collections::HashSet, sync::Arc};
use ui::{Button, ButtonCommon, ButtonSize, Disableable, prelude::*};
use uuid::Uuid;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputViewAction {
    SelectAttempt(AttemptId),
    CancelAttempt(AttemptId),
    InterruptAttempt(AttemptId),
    SelectOutput(Uuid),
    ToggleErrorDetails(AttemptId),
    CopyError(AttemptId),
    NavigateToError(AttemptId),
    CopyReference { output_id: Uuid, reference: String },
    ViewReference { output_id: Uuid, reference: String },
    DownloadReference { output_id: Uuid, reference: String },
    RecoverOutput(Uuid),
    RemoveOutput(Uuid),
}

pub type OutputViewActionHandler = Arc<dyn Fn(OutputViewAction, &mut Window, &mut App) + 'static>;

#[derive(IntoElement)]
pub struct OutputView {
    snapshot: ExecutionSnapshot,
    selected_attempt_id: Option<AttemptId>,
    selected_output_id: Option<Uuid>,
    expanded_error_attempts: HashSet<AttemptId>,
    reference_actions_available: bool,
    output_operations_available: bool,
    runtime_controls_available: bool,
    show_progress: bool,
    selected_view_focus_handle: Option<FocusHandle>,
    on_action: OutputViewActionHandler,
}

impl OutputView {
    pub fn new(
        snapshot: ExecutionSnapshot,
        selected_attempt_id: Option<AttemptId>,
        selected_output_id: Option<Uuid>,
        on_action: OutputViewActionHandler,
    ) -> Self {
        Self {
            snapshot,
            selected_attempt_id,
            selected_output_id,
            expanded_error_attempts: HashSet::new(),
            reference_actions_available: false,
            output_operations_available: false,
            runtime_controls_available: false,
            show_progress: true,
            selected_view_focus_handle: None,
            on_action,
        }
    }

    pub fn with_capabilities(
        mut self,
        reference_actions_available: bool,
        output_operations_available: bool,
    ) -> Self {
        self.reference_actions_available = reference_actions_available;
        self.output_operations_available = output_operations_available;
        self
    }

    pub fn with_expanded_error_attempts(
        mut self,
        expanded_error_attempts: impl IntoIterator<Item = AttemptId>,
    ) -> Self {
        self.expanded_error_attempts = expanded_error_attempts.into_iter().collect();
        self
    }

    pub fn with_show_progress(mut self, show_progress: bool) -> Self {
        self.show_progress = show_progress;
        self
    }

    pub fn with_runtime_controls_available(mut self, available: bool) -> Self {
        self.runtime_controls_available = available;
        self
    }

    pub fn with_selected_view_focus_handle(mut self, focus_handle: FocusHandle) -> Self {
        self.selected_view_focus_handle = Some(focus_handle);
        self
    }
}

impl RenderOnce for OutputView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let attempt_index = self.selected_attempt_id.and_then(|attempt_id| {
            self.snapshot.attempts.iter().position(|attempt| {
                attempt.attempt_id == attempt_id && attempt.profile_id == self.snapshot.profile_id
            })
        });
        let profile_label = self
            .snapshot
            .profile_id
            .0
            .to_string()
            .chars()
            .take(8)
            .collect::<String>();
        let runtime_controls_available =
            self.runtime_controls_available && snapshot_can_show_content(&self.snapshot.status);
        let active_attempts = self
            .snapshot
            .attempts
            .iter()
            .filter(|attempt| !attempt.state.is_terminal())
            .collect::<Vec<_>>();
        let active_attempts_element = (!active_attempts.is_empty()).then(|| {
            render_active_attempts(
                &active_attempts,
                self.selected_attempt_id,
                runtime_controls_available,
                self.on_action.clone(),
                cx,
            )
        });
        let attempt = self
            .snapshot
            .attempts
            .into_iter()
            .enumerate()
            .find_map(|(index, attempt)| (Some(index) == attempt_index).then_some(attempt));

        v_flex()
            .id("sim-execution-output")
            .size_full()
            .min_h_0()
            .bg(cx.theme().colors().panel_background)
            .child(
                h_flex()
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_2()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(div().text_ui_sm(cx).child(format!(
                        "{} execution output · profile {profile_label}",
                        source_name(self.snapshot.source)
                    )))
                    .when_some(attempt.as_ref(), |this, attempt| {
                        this.child(
                            div()
                                .id("sim-output-summary")
                                .role(Role::Status)
                                .aria_label(format!(
                                    "Attempt {}, {}, {} outputs",
                                    attempt.attempt_id.0,
                                    projected_state_name(attempt),
                                    attempt.outputs.len()
                                ))
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(format!(
                                    "Attempt {} · {} · {} outputs",
                                    attempt_label(attempt.attempt_id),
                                    projected_state_name(attempt),
                                    attempt.outputs.len()
                                )),
                        )
                    }),
            )
            .child(snapshot_status(&self.snapshot.status, cx))
            .when_some(active_attempts_element, |this, active_attempts| {
                this.child(active_attempts)
            })
            .when(
                attempt.is_none() && snapshot_can_show_content(&self.snapshot.status),
                |this| {
                    this.child(
                        div()
                            .id("sim-output-no-attempt")
                            .role(Role::Status)
                            .aria_label("No execution attempt selected for output inspection")
                            .flex_1()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_ui_sm(cx)
                            .text_color(cx.theme().colors().text_muted)
                            .child(
                                "Select an attempt to inspect its progress, errors, and outputs",
                            ),
                    )
                },
            )
            .when_some(
                attempt.filter(|_| snapshot_can_show_content(&self.snapshot.status)),
                |this, attempt| {
                    this.child(render_attempt_output(
                        attempt,
                        self.selected_output_id,
                        &self.expanded_error_attempts,
                        self.reference_actions_available,
                        self.output_operations_available,
                        runtime_controls_available,
                        self.show_progress,
                        self.selected_view_focus_handle.as_ref(),
                        self.on_action,
                        matches!(self.snapshot.status, ExecutionSnapshotStatus::Ready),
                        self.snapshot.revision,
                        window,
                        cx,
                    ))
                },
            )
    }
}

fn render_attempt_output(
    mut attempt: AttemptPresentation,
    selected_output_id: Option<Uuid>,
    expanded_error_attempts: &HashSet<AttemptId>,
    reference_actions_available: bool,
    output_operations_available: bool,
    runtime_controls_available: bool,
    show_progress: bool,
    selected_view_focus_handle: Option<&FocusHandle>,
    on_action: OutputViewActionHandler,
    preview_pixels_available: bool,
    snapshot_revision: u64,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let mut outputs = std::mem::take(&mut attempt.outputs);
    outputs.sort_by(|left, right| {
        left.node_id
            .0
            .cmp(&right.node_id.0)
            .then(left.output_index.cmp(&right.output_index))
            .then(left.output_id.cmp(&right.output_id))
    });

    v_flex()
        .id(format!("sim-attempt-output-{}", attempt.attempt_id.0))
        .flex_1()
        .min_h_0()
        .overflow_y_scroll()
        .when(show_progress, |this| {
            this.child(render_attempt_progress(&attempt, cx))
        })
        .when_some(attempt.preview.take(), |this, preview| {
            let preview_identity = preview_identity(preview.frame_index, preview.output_index);
            let accessibility_label = format!(
                "Preview for node {}, revision {}, {}{}, {} encoded bytes",
                preview.node_id.0,
                preview.revision,
                preview.media_type,
                preview_identity,
                preview.encoded_bytes.len(),
            );
            let title = format!(
                "Latest preview · {} · revision {}{}{}",
                preview.media_type,
                preview.revision,
                dimensions(preview.width, preview.height),
                preview_identity,
            );
            let details = format!(
                "{} encoded bytes · node {} · {} preview metadata · {} retained keyed previews",
                preview.encoded_bytes.len(),
                preview.node_id.0,
                media_kind_name(preview.media_kind),
                attempt.previews.len().max(1),
            );
            let preview_id = preview.preview_id;
            let native_preview = if preview_pixels_available {
                render_native_preview(
                    attempt.profile_id,
                    attempt.prompt_id,
                    attempt.attempt_id,
                    preview,
                    window,
                    cx,
                )
            } else {
                native_preview_status("Image preview is unavailable while execution state is stale")
            };
            this.child(
                v_flex()
                    .id(format!("sim-output-preview-{preview_id}"))
                    .role(Role::Status)
                    .aria_label(accessibility_label)
                    .w_full()
                    .px_2()
                    .py_1()
                    .gap_0p5()
                    .border_b_1()
                    .border_color(cx.theme().colors().border_variant)
                    .child(div().text_ui_sm(cx).child(title))
                    .child(
                        div()
                            .text_ui_sm(cx)
                            .text_color(cx.theme().colors().text_muted)
                            .child(details),
                    )
                    .child(native_preview),
            )
        })
        .when_some(attempt.interrupted_reason.as_ref(), |this, reason| {
            this.child(
                div()
                    .id(format!("sim-output-interrupted-{}", attempt.attempt_id.0))
                    .role(Role::Status)
                    .aria_label(format!("Execution interrupted: {reason}"))
                    .w_full()
                    .px_2()
                    .py_1()
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!("Interrupted · {reason}")),
            )
        })
        .when_some(attempt.failure.as_ref(), |this, failure| {
            this.child(render_failure(
                attempt.attempt_id,
                failure,
                expanded_error_attempts.contains(&attempt.attempt_id),
                on_action.clone(),
                cx,
            ))
        })
        .when(outputs.is_empty() && !attempt.state.is_terminal(), |this| {
            this.child(render_in_progress_skeleton(
                &attempt,
                runtime_controls_available,
                on_action.clone(),
                cx,
            ))
        })
        .when(outputs.is_empty() && attempt.state.is_terminal(), |this| {
            this.child(
                div()
                    .id(format!("sim-output-empty-{}", attempt.attempt_id.0))
                    .role(Role::Status)
                    .aria_label(format!(
                        "Attempt {} has no committed outputs",
                        attempt.attempt_id.0
                    ))
                    .w_full()
                    .p_2()
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child("No committed outputs for this attempt"),
            )
        })
        .when_some(
            outputs
                .iter()
                .find(|output| Some(output.output_id) == selected_output_id),
            |this, output| {
                this.child(render_selected_native_image(
                    NativePreviewScope {
                        profile_id: attempt.profile_id,
                        prompt_id: attempt.prompt_id,
                        attempt_id: attempt.attempt_id,
                    },
                    snapshot_revision,
                    output,
                    preview_pixels_available,
                    window,
                    cx,
                ))
            },
        )
        .when(!outputs.is_empty(), |this| {
            this.child(
                v_flex()
                    .id(format!("sim-output-list-{}", attempt.attempt_id.0))
                    .role(Role::List)
                    .aria_label(format!(
                        "Ordered outputs for attempt {}",
                        attempt.attempt_id.0
                    ))
                    .w_full()
                    .children(outputs.iter().map(|output| {
                        let selected = selected_output_id == Some(output.output_id);
                        render_output_row(
                            output,
                            selected,
                            reference_actions_available,
                            output_operations_available,
                            selected.then_some(selected_view_focus_handle).flatten(),
                            on_action.clone(),
                            cx,
                        )
                    })),
            )
        })
        .into_any_element()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativePreviewScope {
    pub profile_id: sim_types::ProfileId,
    pub prompt_id: sim_types::PromptId,
    pub attempt_id: AttemptId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeImageOutputKey {
    output_id: Uuid,
    reference: String,
    byte_length: u64,
    snapshot_revision: u64,
    binding: NativeImageImportBinding,
}

#[derive(Clone)]
enum NativeImageSource {
    Preview(Arc<ExecutionPreview>),
    Output {
        key: NativeImageOutputKey,
        reader: Arc<dyn NativeImageOutputReader>,
    },
}

impl NativeImageSource {
    fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Preview(left), Self::Preview(right)) => left == right,
            (
                Self::Output {
                    key: left,
                    reader: left_reader,
                },
                Self::Output {
                    key: right,
                    reader: right_reader,
                },
            ) => left == right && Arc::ptr_eq(left_reader, right_reader),
            _ => false,
        }
    }

    fn is_current(&self, scope: &NativePreviewScope, cx: &App) -> bool {
        match self {
            Self::Preview(_) => true,
            Self::Output { key, .. } => native_asset_services(cx)
                .and_then(|services| services.image_output_reader())
                .is_some_and(|reader| {
                    reader.binding() == key.binding
                        && key.binding.profile_id() == scope.profile_id.0.to_string()
                }),
        }
    }

    fn prepare(&self, cancellation: &CancellationToken) -> Result<Arc<Image>, SharedString> {
        match self {
            Self::Preview(preview) => prepare_native_preview(preview, cancellation),
            Self::Output { key, reader } => {
                cancellation
                    .check()
                    .map_err(|error| SharedString::from(error.to_string()))?;
                if reader.binding() != key.binding || key.byte_length > reader.maximum_input_bytes()
                {
                    return Err("Image output service binding or byte limit changed".into());
                }
                let bytes = reader
                    .read(&key.reference, cancellation)
                    .map_err(|failure| {
                        SharedString::from(format!("{}: {}", failure.code, failure.message))
                    })?;
                if u64::try_from(bytes.len()).ok() != Some(key.byte_length) {
                    return Err(
                        "Image output byte length does not match its durable metadata".into(),
                    );
                }
                prepare_native_png(bytes, None, None, cancellation)
            }
        }
    }
}

fn render_selected_native_image(
    scope: NativePreviewScope,
    snapshot_revision: u64,
    output: &ExecutionOutput,
    ready_snapshot: bool,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    window.use_keyed_state(
        "sim-output-selected-native-image-binding-state",
        cx,
        |_, cx| {
            cx.observe_global::<crate::GlobalNativeSubgraphCatalogRevision>(
                |_: &mut Subscription, cx| cx.notify(),
            )
        },
    );
    if !ready_snapshot {
        return native_preview_status("Image output is unavailable while execution state is stale");
    }
    if output.media_kind != OutputMediaKind::Image || output.media_type != "image/png" {
        return native_preview_status(
            "This output format is not supported by the native image viewer",
        );
    }
    let ExecutionOutputAvailability::Ready {
        reference,
        byte_length,
    } = &output.availability
    else {
        return native_preview_status("The selected image output is not available for inspection");
    };
    if *byte_length > sim_types::MAX_ENCODED_PREVIEW_BYTES as u64 {
        return native_preview_status("Image output exceeds the viewer's encoded byte limit");
    }
    let Some(reader) =
        native_asset_services(cx).and_then(|services| services.image_output_reader())
    else {
        return native_preview_status("Native image output service is not available");
    };
    let binding = reader.binding();
    if binding.profile_id() != scope.profile_id.0.to_string() {
        return native_preview_status("Image output belongs to a different native profile");
    }
    let source = Arc::new(NativeImageSource::Output {
        key: NativeImageOutputKey {
            output_id: output.output_id,
            reference: reference.clone(),
            byte_length: *byte_length,
            snapshot_revision,
            binding,
        },
        reader,
    });
    let state = native_image_state(
        "sim-output-selected-native-image-state",
        scope,
        source,
        window,
        cx,
    );
    div()
        .id("sim-output-selected-native-image")
        .w_full()
        .child(render_native_image_state(state, cx))
        .into_any_element()
}

pub(super) struct NativePreviewState {
    scope: NativePreviewScope,
    source: Arc<NativeImageSource>,
    cancellation: CancellationToken,
    preparation: Option<Task<()>>,
    pub result: Option<Result<Arc<Image>, SharedString>>,
    _binding_subscription: Subscription,
}

impl NativePreviewState {
    fn clear(&mut self, cx: &mut App) {
        self.cancellation.cancel();
        drop(self.preparation.take());
        if let Some(Ok(image)) = self.result.take() {
            ImageSource::Image(image).remove_asset(cx);
        }
    }

    fn retry(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.result, Some(Err(_))) {
            return;
        }
        if !self.source.is_current(&self.scope, cx) {
            self.clear(cx);
            self.result = Some(Err(
                "Native image output service is no longer available".into()
            ));
            cx.notify();
            return;
        }
        self.clear(cx);
        // A new source identity also rejects a completion queued by the cancelled preparation.
        self.source = Arc::new(self.source.as_ref().clone());
        self.cancellation = CancellationToken::default();
        self.start(cx);
        cx.notify();
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        let source = self.source.clone();
        let scope = self.scope.clone();
        let cancellation = self.cancellation.clone();
        let preparation = cx.background_spawn({
            let source = source.clone();
            let cancellation = cancellation.clone();
            async move { source.prepare(&cancellation) }
        });
        self.preparation = Some(cx.spawn(async move |this, cx| {
            let result = preparation.await;
            if let Err(error) = this.update(cx, |this, cx| {
                if this.scope == scope && Arc::ptr_eq(&this.source, &source) {
                    this.result = Some(if cancellation.is_cancelled() {
                        Err("Image preparation was cancelled".into())
                    } else if !source.is_current(&scope, cx) {
                        Err("Native image output service is no longer available".into())
                    } else {
                        result
                    });
                    cx.notify();
                }
            }) {
                log::debug!("Native image preview view was released: {error}");
            }
        }));
    }
}

impl Drop for NativePreviewState {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

pub(super) fn native_preview_limits() -> PngLimits {
    PngLimits {
        max_input_bytes: sim_types::MAX_ENCODED_PREVIEW_BYTES,
        max_dimension: 1_024,
        max_pixels: 1_048_576,
        max_decoder_allocation_bytes: 16 * 1_024 * 1_024,
    }
}

pub(super) fn prepare_native_preview(
    preview: &ExecutionPreview,
    cancellation: &CancellationToken,
) -> Result<Arc<Image>, SharedString> {
    cancellation
        .check()
        .map_err(|error| SharedString::from(error.to_string()))?;
    if preview.media_kind != OutputMediaKind::Image || preview.media_type != "image/png" {
        return Err("This preview format is not supported by the native image viewer".into());
    }
    prepare_native_png(
        preview.encoded_bytes.clone(),
        preview.width,
        preview.height,
        cancellation,
    )
}

fn prepare_native_png(
    encoded_bytes: Vec<u8>,
    width: Option<u32>,
    height: Option<u32>,
    cancellation: &CancellationToken,
) -> Result<Arc<Image>, SharedString> {
    cancellation
        .check()
        .map_err(|error| SharedString::from(error.to_string()))?;
    let decoded = decode_png(&encoded_bytes, native_preview_limits()).map_err(|error| {
        SharedString::from(format!("Image preview could not be decoded: {error}"))
    })?;
    if width.is_some_and(|width| width != decoded.width)
        || height.is_some_and(|height| height != decoded.height)
    {
        return Err("Image preview dimensions do not match its encoded pixels".into());
    }
    cancellation
        .check()
        .map_err(|error| SharedString::from(error.to_string()))?;
    drop(decoded);
    // The media owner admitted the actual static PNG dimensions before GPUI's cached decode.
    Ok(Arc::new(Image::from_bytes(ImageFormat::Png, encoded_bytes)))
}

pub(super) fn render_native_preview(
    profile_id: sim_types::ProfileId,
    prompt_id: sim_types::PromptId,
    attempt_id: AttemptId,
    preview: ExecutionPreview,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    if preview.encoded_bytes.len() > sim_types::MAX_ENCODED_PREVIEW_BYTES {
        return native_preview_status("Image preview exceeds the encoded byte limit");
    }
    let state = native_image_state(
        "sim-output-native-preview-state",
        NativePreviewScope {
            profile_id,
            prompt_id,
            attempt_id,
        },
        Arc::new(NativeImageSource::Preview(Arc::new(preview))),
        window,
        cx,
    );
    render_native_image_state(state, cx)
}

fn render_native_image_state(state: gpui::Entity<NativePreviewState>, cx: &App) -> AnyElement {
    match state.read(cx).result.as_ref() {
        Some(Ok(image)) => img(image.clone())
            .id("sim-output-native-preview-image")
            .debug_selector(|| "SIM-NATIVE-PREVIEW-IMAGE".into())
            .role(Role::Image)
            .aria_label("Native image preview")
            .w_full()
            .h(gpui::px(256.0))
            .object_fit(ObjectFit::Contain)
            .with_loading(|| native_preview_status("Loading image preview"))
            .with_fallback(|| native_preview_status("Image preview could not be displayed"))
            .into_any_element(),
        Some(Err(error)) => {
            let rendered_scope = state.read(cx).scope.clone();
            let rendered_source = state.read(cx).source.clone();
            let state = state.downgrade();
            v_flex()
                .items_start()
                .child(native_preview_status(error.clone()))
                .child(
                    div()
                        .debug_selector(|| "SIM-NATIVE-IMAGE-RETRY".into())
                        .child(
                            Button::new("sim-native-image-retry", "Retry Image")
                                .size(ButtonSize::Compact)
                                .on_click(move |_, _, cx| {
                                    if let Err(error) = state.update(cx, |state, cx| {
                                        if state.scope == rendered_scope
                                            && Arc::ptr_eq(&state.source, &rendered_source)
                                        {
                                            state.retry(cx);
                                        }
                                    }) {
                                        log::debug!(
                                            "Native image retry view was released: {error}"
                                        );
                                    }
                                }),
                        ),
                )
                .into_any_element()
        }
        None => native_preview_status("Preparing image preview"),
    }
}

#[cfg(all(test, feature = "test-support"))]
pub(super) fn native_preview_state(
    scope: NativePreviewScope,
    source: Arc<ExecutionPreview>,
    window: &mut Window,
    cx: &mut App,
) -> gpui::Entity<NativePreviewState> {
    native_image_state(
        "sim-output-native-preview-state",
        scope,
        Arc::new(NativeImageSource::Preview(source)),
        window,
        cx,
    )
}

fn native_image_state(
    key: &'static str,
    scope: NativePreviewScope,
    source: Arc<NativeImageSource>,
    window: &mut Window,
    cx: &mut App,
) -> gpui::Entity<NativePreviewState> {
    let state = window.use_keyed_state(key, cx, |_, cx| {
        cx.on_release(|this: &mut NativePreviewState, cx| this.clear(cx))
            .detach();
        let binding_subscription = cx.observe_global::<crate::GlobalNativeSubgraphCatalogRevision>(
            |this: &mut NativePreviewState, cx| {
                if !this.source.is_current(&this.scope, cx) {
                    this.clear(cx);
                    this.result = Some(Err(
                        "Native image output service is no longer available".into()
                    ));
                    cx.notify();
                }
            },
        );
        let mut state = NativePreviewState {
            scope: scope.clone(),
            source: source.clone(),
            cancellation: CancellationToken::default(),
            preparation: None,
            result: None,
            _binding_subscription: binding_subscription,
        };
        state.start(cx);
        state
    });
    state.update(cx, |state, cx| {
        if state.scope != scope || !state.source.matches(&source) {
            state.clear(cx);
            state.scope = scope;
            state.source = source;
            state.cancellation = CancellationToken::default();
            state.start(cx);
        }
    });
    state
}

#[cfg(all(test, feature = "test-support"))]
impl NativePreviewState {
    pub(super) fn cancellation_for_test(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}

fn native_preview_status(message: impl Into<SharedString>) -> AnyElement {
    let message = message.into();
    div()
        .id("sim-output-native-preview-status")
        .debug_selector(|| "SIM-NATIVE-PREVIEW-STATUS".into())
        .role(Role::Status)
        .aria_label(message.clone())
        .child(message)
        .into_any_element()
}

fn render_active_attempts(
    attempts: &[&AttemptPresentation],
    selected_attempt_id: Option<AttemptId>,
    runtime_controls_available: bool,
    on_action: OutputViewActionHandler,
    cx: &mut App,
) -> AnyElement {
    v_flex()
        .id("sim-output-history-active-queue-item")
        .debug_selector(|| "SIM-SURFACE-OUTPUT-HISTORY-ACTIVE-QUEUE-ITEM".into())
        .role(Role::List)
        .aria_label(format!(
            "{} in-progress execution items outside output history",
            attempts.len()
        ))
        .w_full()
        .border_b_1()
        .border_color(cx.theme().colors().border)
        .children(attempts.iter().map(|attempt| {
            let attempt_id = attempt.attempt_id;
            let selected = selected_attempt_id == Some(attempt_id);
            let select_handler = on_action.clone();
            let cancel_handler = on_action.clone();
            let action = cancellation_action(attempt);
            let controls_available = runtime_controls_available && action.is_some();
            let action_label = "Cancel";

            h_flex()
                .id(format!(
                    "sim-output-history-active-queue-item-{}",
                    attempt_id.0
                ))
                .role(Role::ListItem)
                .aria_selected(selected)
                .aria_label(format!(
                    "Attempt {}, {}, in progress",
                    attempt_id.0,
                    projected_state_name(attempt)
                ))
                .w_full()
                .px_2()
                .py_1()
                .gap_2()
                .justify_between()
                .when(selected, |this| {
                    this.bg(cx.theme().colors().element_selected)
                })
                .child(
                    Button::new(
                        format!("sim-output-history-select-{}", attempt_id.0),
                        format!(
                            "Attempt {} · {}",
                            attempt_label(attempt_id),
                            projected_state_name(attempt)
                        ),
                    )
                    .size(ButtonSize::Compact)
                    .toggle_state(selected)
                    .aria_label(format!(
                        "Inspect in-progress execution attempt {}",
                        attempt_id.0
                    ))
                    .on_click(move |_, window, cx| {
                        select_handler(OutputViewAction::SelectAttempt(attempt_id), window, cx);
                    }),
                )
                .child(
                    Button::new(
                        format!("sim-output-history-cancel-{}", attempt_id.0),
                        action_label,
                    )
                    .size(ButtonSize::Compact)
                    .disabled(!controls_available)
                    .aria_label(if !runtime_controls_available {
                        format!(
                            "{action_label} unavailable: native runtime controller is not connected or snapshot is not ready"
                        )
                    } else if action.is_none() {
                        "Cancellation request is already pending".to_owned()
                    } else if attempt.state == AttemptState::Running {
                        format!(
                            "Cancel running execution attempt {} by sending an interrupt request",
                            attempt_id.0
                        )
                    } else {
                        format!("{action_label} execution attempt {}", attempt_id.0)
                    })
                    .on_click(move |_, window, cx| {
                        if let Some(action) = action.clone() {
                            cancel_handler(action, window, cx);
                        }
                    }),
                )
        }))
        .into_any_element()
}

fn render_in_progress_skeleton(
    attempt: &AttemptPresentation,
    runtime_controls_available: bool,
    on_action: OutputViewActionHandler,
    cx: &mut App,
) -> AnyElement {
    let attempt_id = attempt.attempt_id;
    let action = cancellation_action(attempt);
    let controls_available = runtime_controls_available && action.is_some();
    let action_label = "Cancel";

    v_flex()
        .id(format!(
            "sim-output-history-skeleton-{}",
            attempt_id.0
        ))
        .debug_selector(|| "SIM-OUTPUT-HISTORY-SKELETON".into())
        .role(Role::Status)
        .aria_label(format!(
            "Output skeleton for in-progress attempt {}, {}",
            attempt_id.0,
            projected_state_name(attempt)
        ))
        .w_full()
        .p_2()
        .gap_2()
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .child(
                    div()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child("Generating native output…"),
                )
                .child(
                    Button::new(
                        format!("sim-output-skeleton-cancel-{}", attempt_id.0),
                        action_label,
                    )
                    .size(ButtonSize::Compact)
                    .disabled(!controls_available)
                    .aria_label(if !runtime_controls_available {
                        format!(
                            "{action_label} unavailable: native runtime controller is not connected or snapshot is not ready"
                        )
                    } else if action.is_none() {
                        "Cancellation request is already pending".to_owned()
                    } else if attempt.state == AttemptState::Running {
                        format!(
                            "Cancel running execution attempt {} by sending an interrupt request",
                            attempt_id.0
                        )
                    } else {
                        format!("{action_label} execution attempt {}", attempt_id.0)
                    })
                    .on_click(move |_, window, cx| {
                        if let Some(action) = action.clone() {
                            on_action(action, window, cx);
                        }
                    }),
                ),
        )
        .child(
            div()
                .id(format!("sim-output-skeleton-preview-{}", attempt_id.0))
                .h_24()
                .w_full()
                .rounded_sm()
                .bg(cx.theme().colors().element_background),
        )
        .child(
            div()
                .id(format!("sim-output-skeleton-caption-{}", attempt_id.0))
                .h_2()
                .w_1_2()
                .rounded_full()
                .bg(cx.theme().colors().element_background),
        )
        .into_any_element()
}

fn cancellation_action(attempt: &AttemptPresentation) -> Option<OutputViewAction> {
    match attempt.state {
        AttemptState::Queued => Some(OutputViewAction::CancelAttempt(attempt.attempt_id)),
        AttemptState::Running => Some(OutputViewAction::InterruptAttempt(attempt.attempt_id)),
        AttemptState::Cancelling
        | AttemptState::Succeeded
        | AttemptState::Failed
        | AttemptState::Cancelled
        | AttemptState::Interrupted => None,
    }
}

fn render_attempt_progress(attempt: &AttemptPresentation, cx: &mut App) -> AnyElement {
    let Some(progress) = attempt.progress.as_ref() else {
        return div().into_any_element();
    };
    let total = progress.total.max(1);
    let completed = progress.completed.min(total);
    let attempt_label = attempt_label(attempt.attempt_id);

    v_flex()
        .id(format!("sim-output-progress-{}", attempt.attempt_id.0))
        .role(Role::ProgressIndicator)
        .aria_label(progress.node_id.as_ref().map_or_else(
            || format!("Attempt {attempt_label} progress"),
            |node_id| format!("Attempt {attempt_label} progress for node {}", node_id.0),
        ))
        .aria_value(format!("{completed} of {total}"))
        .aria_numeric_value(completed as f64)
        .aria_min_numeric_value(0.0)
        .aria_max_numeric_value(total as f64)
        .w_full()
        .px_2()
        .py_1()
        .gap_0p5()
        .border_b_1()
        .border_color(cx.theme().colors().border_variant)
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .child(div().text_ui_sm(cx).child("Progress"))
                .child(
                    div()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child(format!("{completed} / {total}")),
                ),
        )
        .child(
            div()
                .h_1()
                .w_full()
                .rounded_full()
                .bg(cx.theme().colors().element_background)
                .child(
                    div()
                        .h_full()
                        .rounded_full()
                        .bg(cx.theme().status().info)
                        .w(relative(completed as f32 / total as f32)),
                ),
        )
        .into_any_element()
}

fn render_output_row(
    output: &ExecutionOutput,
    selected: bool,
    reference_actions_available: bool,
    output_operations_available: bool,
    selected_view_focus_handle: Option<&FocusHandle>,
    on_action: OutputViewActionHandler,
    cx: &mut App,
) -> AnyElement {
    let output_id = output.output_id;
    let output_label = output_label(output_id);
    let availability_label = availability_name(&output.availability);
    let recovery = output.recovery_eligibility();
    let removal = output.removal_eligibility();
    let recovery_allowed = output_operations_available && recovery.is_allowed();
    let removal_allowed = output_operations_available && removal.is_allowed();
    let recovery_reason = if output_operations_available {
        eligibility_reason(&recovery)
    } else {
        "no native profile artifact service is registered"
    };
    let removal_reason = if output_operations_available {
        eligibility_reason(&removal)
    } else {
        "no native profile artifact service is registered"
    };
    let reference_retained = !matches!(
        &output.availability,
        ExecutionOutputAvailability::Removed { .. }
    );
    let copy_reference = reference_retained
        .then(|| output_reference(output).map(str::to_owned))
        .flatten();
    let view_reference = reference_retained
        .then(|| output.view_reference.clone())
        .flatten();
    let download_reference = reference_retained
        .then(|| output.download_reference.clone())
        .flatten();
    let select_handler = on_action.clone();
    let view_handler = on_action.clone();
    let keyboard_view_handler = on_action.clone();
    let keyboard_view_reference = view_reference.clone();
    let download_handler = on_action.clone();
    let copy_handler = on_action.clone();
    let recover_handler = on_action.clone();
    let remove_handler = on_action;
    let output_aria_label = format!(
        "Output {}, node {}, port {}, {}, media type {}, {}, subfolder {}, storage type {}, {} metadata fields, view reference {}, download reference {}",
        output.name,
        output.node_id.0,
        output.output_index,
        media_kind_name(output.media_kind),
        output.media_type,
        availability_label,
        optional_label(output.subfolder.as_deref()),
        optional_label(output.storage_type.as_deref()),
        output.metadata.len(),
        optional_label(output.view_reference.as_deref()),
        optional_label(output.download_reference.as_deref())
    );

    v_flex()
        .id(format!("sim-output-row-{output_id}"))
        .role(Role::ListItem)
        .aria_selected(selected)
        .aria_label(output_aria_label)
        .w_full()
        .px_2()
        .py_1()
        .gap_1()
        .border_b_1()
        .border_color(cx.theme().colors().border_variant)
        .when(selected, |this| {
            this.bg(cx.theme().colors().element_selected)
        })
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .gap_2()
                .child(
                    v_flex()
                        .min_w_0()
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .child(format!("{} · {}", output.name, output_label)),
                        )
                        .child(
                            div()
                                .text_ui_sm(cx)
                                .text_color(cx.theme().colors().text_muted)
                                .child(format!(
                                    "Node {} · port {} · {} · {}",
                                    output.node_id.0,
                                    output.output_index,
                                    media_kind_name(output.media_kind),
                                    output.media_type
                                )),
                        ),
                )
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            div()
                                .debug_selector(|| "SIM-OUTPUT-INSPECT".into())
                                .child(
                                    Button::new(
                                        format!("sim-select-output-{output_id}"),
                                        "Inspect",
                                    )
                                    .size(ButtonSize::Compact)
                                    .aria_label(format!("Inspect output {}", output.name))
                                    .on_click(move |_, window, cx| {
                                        select_handler(
                                            OutputViewAction::SelectOutput(output_id),
                                            window,
                                            cx,
                                        );
                                    }),
                                ),
                        )
                        .child(
                            div()
                                .id(format!("sim-output-view-keyboard-{output_id}"))
                                .role(Role::Button)
                                .when_some(
                                    selected_view_focus_handle.cloned(),
                                    move |this, focus_handle| {
                                        this.tab_stop(true)
                                            .tab_index(0)
                                            .track_focus(&focus_handle)
                                            .on_key_down(
                                                move |event: &KeyDownEvent, window, cx| {
                                                    if matches!(
                                                        event.keystroke.key.as_str(),
                                                        "enter" | "space"
                                                    ) && reference_actions_available
                                                        && let Some(reference) =
                                                            keyboard_view_reference.clone()
                                                    {
                                                        cx.stop_propagation();
                                                        keyboard_view_handler(
                                                            OutputViewAction::ViewReference {
                                                                output_id,
                                                                reference,
                                                            },
                                                            window,
                                                            cx,
                                                        );
                                                    }
                                                },
                                            )
                                    },
                                )
                                .debug_selector(|| "SIM-OUTPUT-VIEW".into())
                                .child(
                                    Button::new(format!("sim-view-output-{output_id}"), "View")
                                        .size(ButtonSize::Compact)
                                        .disabled(
                                            view_reference.is_none()
                                                || !reference_actions_available,
                                        )
                                        .aria_label(view_reference.as_ref().map_or_else(
                                            || format!("View output {} unavailable: no view reference", output.name),
                                            |_| {
                                                if reference_actions_available {
                                                    format!("View output {} using its declared view reference", output.name)
                                                } else {
                                                    format!("View output {} unavailable: no native reference handler is registered", output.name)
                                                }
                                            },
                                        ))
                                        .on_click(move |_, window, cx| {
                                            if let Some(reference) = view_reference.clone() {
                                                view_handler(
                                                    OutputViewAction::ViewReference {
                                                        output_id,
                                                        reference,
                                                    },
                                                    window,
                                                    cx,
                                                );
                                            }
                                        }),
                                ),
                        )
                        .child(
                            Button::new(
                                format!("sim-download-output-{output_id}"),
                                "Download",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(
                                download_reference.is_none() || !reference_actions_available,
                            )
                            .aria_label(download_reference.as_ref().map_or_else(
                                || {
                                    format!(
                                        "Download output {} unavailable: no download reference",
                                        output.name
                                    )
                                },
                                |_| {
                                    if reference_actions_available {
                                        format!(
                                            "Download output {} using its declared download reference",
                                            output.name
                                        )
                                    } else {
                                        format!(
                                            "Download output {} unavailable: no native reference handler is registered",
                                            output.name
                                        )
                                    }
                                },
                            ))
                            .on_click(move |_, window, cx| {
                                if let Some(reference) = download_reference.clone() {
                                    download_handler(
                                        OutputViewAction::DownloadReference {
                                            output_id,
                                            reference,
                                        },
                                        window,
                                        cx,
                                    );
                                }
                                }),
                        )
                        .child(
                            Button::new(format!("sim-copy-output-{output_id}"), "Copy")
                                .size(ButtonSize::Compact)
                                .disabled(copy_reference.is_none())
                                .aria_label(copy_reference.as_ref().map_or_else(
                                    || format!("Copy output {} reference unavailable", output.name),
                                    |_| format!("Copy reference for output {}", output.name),
                                ))
                                .on_click(move |_, window, cx| {
                                    if let Some(reference) = copy_reference.clone() {
                                        copy_handler(
                                            OutputViewAction::CopyReference {
                                                output_id,
                                                reference,
                                            },
                                            window,
                                            cx,
                                        );
                                    }
                                }),
                        )
                        .child(
                            Button::new(format!("sim-recover-output-{output_id}"), "Recover")
                                .size(ButtonSize::Compact)
                                .disabled(!recovery_allowed)
                                .aria_label(if recovery_allowed {
                                    format!("Recover output {}", output.name)
                                } else {
                                    format!(
                                        "Recover output {} unavailable: {recovery_reason}",
                                        output.name
                                    )
                                })
                                .on_click(move |_, window, cx| {
                                    recover_handler(
                                        OutputViewAction::RecoverOutput(output_id),
                                        window,
                                        cx,
                                    );
                                }),
                        )
                        .child(
                            Button::new(format!("sim-remove-output-{output_id}"), "Remove")
                                .size(ButtonSize::Compact)
                                .disabled(!removal_allowed)
                                .aria_label(if removal_allowed {
                                    format!("Remove output {}", output.name)
                                } else {
                                    format!(
                                        "Remove output {} unavailable: {removal_reason}",
                                        output.name
                                    )
                                })
                                .on_click(move |_, window, cx| {
                                    remove_handler(
                                        OutputViewAction::RemoveOutput(output_id),
                                        window,
                                        cx,
                                    );
                                }),
                        ),
                ),
        )
        .child(render_output_metadata(output, cx))
        .child(render_availability(output, cx))
        .into_any_element()
}

fn render_output_metadata(output: &ExecutionOutput, cx: &mut App) -> AnyElement {
    let subfolder = optional_label(output.subfolder.as_deref());
    let storage_type = optional_label(output.storage_type.as_deref());
    let view_reference = optional_label(output.view_reference.as_deref());
    let download_reference = optional_label(output.download_reference.as_deref());

    v_flex()
        .id(format!("sim-output-metadata-{}", output.output_id))
        .role(Role::List)
        .aria_label(format!(
            "Output metadata: subfolder {subfolder}, storage type {storage_type}, view reference {view_reference}, download reference {download_reference}, {} additional fields",
            output.metadata.len()
        ))
        .w_full()
        .gap_0p5()
        .child(
            div()
                .id(format!("sim-output-subfolder-{}", output.output_id))
                .role(Role::ListItem)
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(format!("Subfolder · {subfolder}")),
        )
        .child(
            div()
                .id(format!("sim-output-storage-type-{}", output.output_id))
                .role(Role::ListItem)
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(format!("Storage type · {storage_type}")),
        )
        .child(
            div()
                .id(format!("sim-output-view-reference-{}", output.output_id))
                .role(Role::ListItem)
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(format!("View reference · {view_reference}")),
        )
        .child(
            div()
                .id(format!(
                    "sim-output-download-reference-{}",
                    output.output_id
                ))
                .role(Role::ListItem)
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(format!("Download reference · {download_reference}")),
        )
        .children(output.metadata.iter().enumerate().map(|(index, (key, value))| {
            div()
                .id(format!("sim-output-metadata-{}-{index}", output.output_id))
                .role(Role::ListItem)
                .aria_label(format!("Output metadata {key}: {value}"))
                .text_ui_sm(cx)
                .text_color(cx.theme().colors().text_muted)
                .child(format!("{key} · {value}"))
        }))
        .into_any_element()
}

fn render_availability(output: &ExecutionOutput, cx: &mut App) -> AnyElement {
    let (role, label, message) = match &output.availability {
        ExecutionOutputAvailability::Ready {
            reference,
            byte_length,
        } => (
            Role::Status,
            format!(
                "Output {} is ready, {} bytes, reference {}",
                output.name, byte_length, reference
            ),
            format!("Ready · {byte_length} bytes · {reference}"),
        ),
        ExecutionOutputAvailability::Missing { reference, reason } => (
            Role::Alert,
            format!("Output {} is missing: {reason}", output.name),
            reference.as_ref().map_or_else(
                || format!("Missing · {reason}"),
                |reference| format!("Missing · {reason} · {reference}"),
            ),
        ),
        ExecutionOutputAvailability::Forbidden { reason } => (
            Role::Alert,
            format!("Output {} is forbidden: {reason}", output.name),
            format!("Forbidden · {reason}"),
        ),
        ExecutionOutputAvailability::Unsupported { media_type, reason } => (
            Role::Alert,
            format!(
                "Output {} has unsupported media type {media_type}: {reason}",
                output.name
            ),
            format!("Unsupported · {media_type} · {reason}"),
        ),
        ExecutionOutputAvailability::Corrupt { reference, reason } => (
            Role::Alert,
            format!("Output {} is corrupt: {reason}", output.name),
            reference.as_ref().map_or_else(
                || format!("Corrupt · {reason}"),
                |reference| format!("Corrupt · {reason} · {reference}"),
            ),
        ),
        ExecutionOutputAvailability::Expired {
            reference,
            expired_at,
            reason,
        } => (
            Role::Alert,
            format!(
                "Output {} expired at {}: {reason}",
                output.name,
                expired_at.to_rfc3339()
            ),
            reference.as_ref().map_or_else(
                || format!("Expired · {} · {reason}", expired_at.to_rfc3339()),
                |reference| {
                    format!(
                        "Expired · {} · {reason} · {reference}",
                        expired_at.to_rfc3339()
                    )
                },
            ),
        ),
        ExecutionOutputAvailability::ExternallyDeleted {
            reference,
            detected_at,
        } => (
            Role::Alert,
            format!(
                "Output {} was deleted outside Zed and detected at {}",
                output.name,
                detected_at.to_rfc3339()
            ),
            format!(
                "Externally deleted · {} · {reference}",
                detected_at.to_rfc3339()
            ),
        ),
        ExecutionOutputAvailability::Removed { reason } => (
            Role::Status,
            format!("Output {} was removed: {reason}", output.name),
            format!("Removed · {reason}"),
        ),
    };

    div()
        .id(format!("sim-output-availability-{}", output.output_id))
        .role(role)
        .aria_label(label)
        .w_full()
        .text_ui_sm(cx)
        .text_color(cx.theme().colors().text_muted)
        .child(message)
        .into_any_element()
}

fn render_failure(
    attempt_id: AttemptId,
    failure: &ExecutionFailure,
    expanded: bool,
    on_action: OutputViewActionHandler,
    cx: &mut App,
) -> AnyElement {
    let detail_summary = failure
        .details
        .iter()
        .map(|(key, value)| format!("{key}: {value}"))
        .collect::<Vec<_>>()
        .join("; ");
    let aria_details = if detail_summary.is_empty() {
        String::new()
    } else {
        format!(". Details: {detail_summary}")
    };
    let toggle_handler = on_action.clone();
    let copy_handler = on_action.clone();
    let navigate_handler = on_action;

    v_flex()
        .id(format!("sim-output-error-{}", attempt_id.0))
        .role(Role::Alert)
        .aria_label(format!(
            "Execution {:?} error {}: {}{}. Retryable: {}",
            failure.origin, failure.code, failure.message, aria_details, failure.retryable
        ))
        .w_full()
        .px_2()
        .py_1()
        .gap_0p5()
        .border_b_1()
        .border_color(cx.theme().status().error_border)
        .bg(cx.theme().status().error_background.opacity(0.16))
        .child(
            h_flex()
                .w_full()
                .justify_between()
                .gap_2()
                .child(div().min_w_0().text_ui_sm(cx).child(format!(
                    "{:?} · {} · {}",
                    failure.origin, failure.code, failure.message
                )))
                .child(
                    h_flex()
                        .gap_1()
                        .child(
                            Button::new(
                                format!("sim-output-error-details-{}", attempt_id.0),
                                if expanded { "Hide Details" } else { "Details" },
                            )
                            .size(ButtonSize::Compact)
                            .aria_expanded(expanded)
                            .aria_label(if expanded {
                                "Hide structured execution error details"
                            } else {
                                "Show structured execution error details"
                            })
                            .on_click(move |_, window, cx| {
                                toggle_handler(
                                    OutputViewAction::ToggleErrorDetails(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        )
                        .child(
                            Button::new(format!("sim-output-copy-error-{}", attempt_id.0), "Copy")
                                .size(ButtonSize::Compact)
                                .aria_label("Copy structured execution error")
                                .on_click(move |_, window, cx| {
                                    copy_handler(
                                        OutputViewAction::CopyError(attempt_id),
                                        window,
                                        cx,
                                    );
                                }),
                        )
                        .child(
                            Button::new(
                                format!("sim-output-locate-error-{}", attempt_id.0),
                                "Locate",
                            )
                            .size(ButtonSize::Compact)
                            .disabled(failure.node_id.is_none())
                            .aria_label(if failure.node_id.is_some() {
                                "Select and reveal the graph node for this execution error"
                            } else {
                                "Locate execution error unavailable: no affected graph node"
                            })
                            .on_click(move |_, window, cx| {
                                navigate_handler(
                                    OutputViewAction::NavigateToError(attempt_id),
                                    window,
                                    cx,
                                );
                            }),
                        ),
                ),
        )
        .when(expanded, |this| {
            this.when_some(failure.node_id.as_ref(), |this, node_id| {
                this.child(
                    div()
                        .text_ui_sm(cx)
                        .text_color(cx.theme().colors().text_muted)
                        .child(format!("Node {}", node_id.0)),
                )
            })
            .children(failure.details.iter().map(|(key, value)| {
                div()
                    .text_ui_sm(cx)
                    .text_color(cx.theme().colors().text_muted)
                    .child(format!("{key}: {value}"))
            }))
        })
        .into_any_element()
}

fn snapshot_status(status: &ExecutionSnapshotStatus, cx: &mut App) -> AnyElement {
    match status {
        ExecutionSnapshotStatus::Loading => div()
            .id("sim-output-loading")
            .role(Role::Status)
            .aria_label("Loading native execution output metadata")
            .w_full()
            .p_2()
            .text_ui_sm(cx)
            .text_color(cx.theme().colors().text_muted)
            .child("Loading output metadata…")
            .into_any_element(),
        ExecutionSnapshotStatus::Ready => div().into_any_element(),
        ExecutionSnapshotStatus::Stale {
            source_revision,
            failure,
        } => {
            let revision = source_revision.map_or_else(
                || "unknown source revision".to_owned(),
                |revision| format!("source revision {revision}"),
            );
            div()
                .id("sim-output-stale")
                .role(Role::Alert)
                .aria_label(format!(
                    "Execution output metadata is stale at {revision}: {}: {}. The last known output metadata remains available for inspection",
                    failure.code, failure.message
                ))
                .w_full()
                .p_2()
                .bg(cx.theme().status().warning_background.opacity(0.2))
                .text_ui_sm(cx)
                .child(format!(
                    "Stale output data · {revision} · {}: {} · showing the last known outputs",
                    failure.code, failure.message
                ))
                .into_any_element()
        }
        ExecutionSnapshotStatus::Partial { failure } => div()
            .id("sim-output-partial")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution output metadata is partially available: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().warning_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "Partial output data · {}: {}",
                failure.code, failure.message
            ))
            .into_any_element(),
        ExecutionSnapshotStatus::Unavailable { failure } => div()
            .id("sim-output-unavailable")
            .role(Role::Alert)
            .aria_label(format!(
                "Execution output metadata is unavailable: {}: {}",
                failure.code, failure.message
            ))
            .w_full()
            .p_2()
            .bg(cx.theme().status().error_background.opacity(0.2))
            .text_ui_sm(cx)
            .child(format!(
                "Output metadata unavailable · {}: {}",
                failure.code, failure.message
            ))
            .into_any_element(),
    }
}

fn snapshot_can_show_content(status: &ExecutionSnapshotStatus) -> bool {
    matches!(
        status,
        ExecutionSnapshotStatus::Ready
            | ExecutionSnapshotStatus::Stale { .. }
            | ExecutionSnapshotStatus::Partial { .. }
    )
}

fn availability_name(availability: &ExecutionOutputAvailability) -> &'static str {
    match availability {
        ExecutionOutputAvailability::Ready { .. } => "ready",
        ExecutionOutputAvailability::Missing { .. } => "missing",
        ExecutionOutputAvailability::Forbidden { .. } => "forbidden",
        ExecutionOutputAvailability::Unsupported { .. } => "unsupported",
        ExecutionOutputAvailability::Corrupt { .. } => "corrupt",
        ExecutionOutputAvailability::Expired { .. } => "expired",
        ExecutionOutputAvailability::ExternallyDeleted { .. } => "externally deleted",
        ExecutionOutputAvailability::Removed { .. } => "removed",
    }
}

fn output_reference(output: &ExecutionOutput) -> Option<&str> {
    output
        .download_reference
        .as_deref()
        .or(output.view_reference.as_deref())
        .or_else(|| match &output.availability {
            ExecutionOutputAvailability::Ready { reference, .. } => Some(reference),
            ExecutionOutputAvailability::Missing { reference, .. }
            | ExecutionOutputAvailability::Corrupt { reference, .. }
            | ExecutionOutputAvailability::Expired { reference, .. } => reference.as_deref(),
            ExecutionOutputAvailability::ExternallyDeleted { reference, .. } => Some(reference),
            ExecutionOutputAvailability::Forbidden { .. }
            | ExecutionOutputAvailability::Unsupported { .. }
            | ExecutionOutputAvailability::Removed { .. } => None,
        })
}

fn eligibility_reason(eligibility: &OperationEligibility) -> &str {
    match eligibility {
        OperationEligibility::Allowed => "available",
        OperationEligibility::Unavailable { reason } => reason,
    }
}

fn media_kind_name(media_kind: OutputMediaKind) -> &'static str {
    match media_kind {
        OutputMediaKind::Image => "image",
        OutputMediaKind::Animation => "animation",
        OutputMediaKind::Video => "video",
        OutputMediaKind::Audio => "audio",
        OutputMediaKind::ThreeD => "3D",
        OutputMediaKind::Text => "text",
        OutputMediaKind::Json => "JSON",
        OutputMediaKind::Binary => "binary",
        OutputMediaKind::Unknown => "unknown",
    }
}

fn source_name(source: ExecutionDataSource) -> &'static str {
    match source {
        ExecutionDataSource::Live => "Live",
        ExecutionDataSource::Persisted => "Persisted",
        ExecutionDataSource::Recovery => "Recovered",
    }
}

fn projected_state_name(attempt: &AttemptPresentation) -> String {
    match attempt.source_projection.as_ref() {
        None => format!("{:?}", attempt.state).to_lowercase(),
        Some(sim_runtime::AttemptSourceProjection::Provider { provider_id, state }) => {
            format!("provider {provider_id}: {}", provider_state_name(state))
        }
        Some(sim_runtime::AttemptSourceProjection::Unknown {
            source_id,
            raw_state,
        }) => source_id.as_ref().map_or_else(
            || format!("unknown source state: {raw_state}"),
            |source_id| format!("unknown source {source_id}: {raw_state}"),
        ),
    }
}

fn provider_state_name(state: &sim_runtime::ProviderAttemptState) -> &str {
    match state {
        sim_runtime::ProviderAttemptState::Queued => "queued",
        sim_runtime::ProviderAttemptState::Running => "running",
        sim_runtime::ProviderAttemptState::Cancelling => "cancelling",
        sim_runtime::ProviderAttemptState::Succeeded => "succeeded",
        sim_runtime::ProviderAttemptState::Failed => "failed",
        sim_runtime::ProviderAttemptState::Cancelled => "cancelled",
        sim_runtime::ProviderAttemptState::Interrupted => "interrupted",
        sim_runtime::ProviderAttemptState::Unknown { raw_state } => raw_state,
    }
}

fn attempt_label(attempt_id: AttemptId) -> SharedString {
    attempt_id
        .0
        .to_string()
        .chars()
        .take(8)
        .collect::<String>()
        .into()
}

fn output_label(output_id: Uuid) -> SharedString {
    output_id
        .to_string()
        .chars()
        .take(8)
        .collect::<String>()
        .into()
}

fn dimensions(width: Option<u32>, height: Option<u32>) -> String {
    match (width, height) {
        (Some(width), Some(height)) => format!(" · {width}×{height}"),
        _ => String::new(),
    }
}

fn preview_identity(frame_index: Option<u64>, output_index: Option<usize>) -> String {
    match (frame_index, output_index) {
        (Some(frame_index), Some(output_index)) => {
            format!(" · frame {frame_index} · output {output_index}")
        }
        (Some(frame_index), None) => format!(" · frame {frame_index}"),
        (None, Some(output_index)) => format!(" · output {output_index}"),
        (None, None) => String::new(),
    }
}

fn optional_label(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}
