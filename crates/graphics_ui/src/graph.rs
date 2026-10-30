use editor::Editor;
use gpui::{
    App, AppContext as _, Context, Entity, EventEmitter, FocusHandle, Focusable, Render,
    Subscription, Window, actions,
};
use ui::{ContextMenu, DropdownMenu, TintColor, prelude::*};
use workspace::{Workspace, item::Item};

use crate::session::{AddMode, GraphicsSession};

actions!(
    graphite,
    [
        /// Opens native Graphite Add, Multiply, Sort, Invert and Color Mix evaluations.
        OpenGraphite,
        /// Focuses the next control in the current Graphite Artboard.
        FocusNextArtboardControl,
        /// Focuses the previous control in the current Graphite Artboard.
        FocusPreviousArtboardControl
    ]
);

pub struct GraphicsWorkspaceItem {
    pub(crate) session: Entity<GraphicsSession>,
    pub(crate) augend_editor: Entity<Editor>,
    pub(crate) addend_editor: Entity<Editor>,
    pub(crate) augend_y_editor: Entity<Editor>,
    pub(crate) addend_y_editor: Entity<Editor>,
    pub(crate) mix_opacity_editor: Entity<Editor>,
    displayed_mix_opacity: String,
    mix_opacity_has_draft: bool,
    mix_opacity_diagnostic: Option<String>,
    native_document: bool,
    numeric_revisions: [u64; 2],
    numeric_diagnostic: Option<String>,
    numeric_drafts: [bool; 2],
    numeric_input_diagnostics: [Option<String>; 2],
    _subscriptions: Vec<Subscription>,
}

impl GraphicsWorkspaceItem {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let session = cx.new(|_| GraphicsSession::new(2.0, 3.0));
        Self::with_session(session, "2".into(), "3".into(), window, cx)
    }

    pub fn from_session(
        session: Entity<GraphicsSession>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<Self, graphics_core::document::NumericDocumentError> {
        let (augend, addend) = session
            .read(cx)
            .numeric_document()
            .ok_or(graphics_core::document::NumericDocumentError::InvalidNodeContract)?
            .inputs()?;
        Ok(Self::with_session(
            session,
            augend.to_string(),
            addend.to_string(),
            window,
            cx,
        ))
    }

    fn with_session(
        session: Entity<GraphicsSession>,
        augend: String,
        addend: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let revision = session
            .read(cx)
            .numeric_document()
            .map_or(0, |document| document.revision());
        let augend_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(augend, window, cx);
            editor.set_placeholder_text("augend", window, cx);
            editor
        });
        let addend_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(addend, window, cx);
            editor.set_placeholder_text("addend", window, cx);
            editor
        });
        let augend_y_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text("0", window, cx);
            editor.set_placeholder_text("augend.y", window, cx);
            editor
        });
        let addend_y_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text("0", window, cx);
            editor.set_placeholder_text("addend.y", window, cx);
            editor
        });
        let mix_opacity_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text("100", window, cx);
            editor.set_placeholder_text("Opacity (%)", window, cx);
            editor
        });
        let mut subscriptions = Vec::new();
        for (editor, input) in [
            (
                &augend_editor,
                Some(graphics_core::document::NumericAddInput::Augend),
            ),
            (
                &addend_editor,
                Some(graphics_core::document::NumericAddInput::Addend),
            ),
            (&augend_y_editor, None),
            (&addend_y_editor, None),
        ] {
            subscriptions.push(cx.subscribe(editor, move |this, editor, event, cx| {
                if matches!(event, editor::EditorEvent::BufferEdited) {
                    if this.session.read(cx).numeric_document().is_some() {
                        if let Some(input) = input {
                            let index = match input {
                                graphics_core::document::NumericAddInput::Augend => 0,
                                graphics_core::document::NumericAddInput::Addend => 1,
                            };
                            let Some(expected_revision) =
                                this.numeric_revisions.get(index).copied()
                            else {
                                return;
                            };
                            let expression = editor.read(cx).text(cx);
                            let canonical = match this
                                .session
                                .read(cx)
                                .numeric_document()
                                .map(|document| document.inputs())
                            {
                                Some(Ok((augend, addend))) => match input {
                                    graphics_core::document::NumericAddInput::Augend => augend,
                                    graphics_core::document::NumericAddInput::Addend => addend,
                                }
                                .to_string(),
                                Some(Err(error)) => {
                                    this.numeric_diagnostic = Some(error.to_string());
                                    cx.notify();
                                    return;
                                }
                                None => return,
                            };
                            // Queued canonical display events are not source commands; reparsing -0 changes its bits.
                            if this.numeric_drafts.get(index) == Some(&false)
                                && expression == canonical
                            {
                                return;
                            }
                            let result = this.session.update(cx, |session, cx| {
                                session.submit_numeric_input(
                                    expected_revision,
                                    input,
                                    &expression,
                                    cx,
                                )
                            });
                            if let Some(draft) = this.numeric_drafts.get_mut(index) {
                                *draft = result.is_err();
                            }
                            if let Some(diagnostic) = this.numeric_input_diagnostics.get_mut(index)
                            {
                                *diagnostic = result.as_ref().err().cloned();
                            }
                            if result.is_ok() {
                                this.numeric_diagnostic = None;
                                if let Some(document) = this.session.read(cx).numeric_document()
                                    && let Some(expected_revision) =
                                        this.numeric_revisions.get_mut(index)
                                {
                                    *expected_revision = document.revision();
                                }
                            }
                            cx.notify();
                        }
                        return;
                    }
                    let augend = this.augend_editor.read(cx).text(cx);
                    let addend = this.addend_editor.read(cx).text(cx);
                    let augend_y = this.augend_y_editor.read(cx).text(cx);
                    let addend_y = this.addend_y_editor.read(cx).text(cx);
                    this.session.update(cx, |session, cx| {
                        session.update_components(&augend, &augend_y, &addend, &addend_y);
                        cx.notify();
                    });
                }
            }));
        }
        subscriptions.push(cx.subscribe_in(
            &mix_opacity_editor,
            window,
            |item, editor, event, window, cx| match event {
                editor::EditorEvent::BufferEdited => {
                    item.mix_opacity_has_draft =
                        editor.read(cx).text(cx) != item.displayed_mix_opacity;
                    cx.notify();
                }
                editor::EditorEvent::Blurred => item.commit_mix_opacity(window, cx),
                _ => {}
            },
        ));
        subscriptions.push(cx.observe(&session, |_, _, cx| cx.notify()));
        let native_document = session.read(cx).numeric_document().is_some();
        Self {
            session,
            augend_editor,
            addend_editor,
            augend_y_editor,
            addend_y_editor,
            mix_opacity_editor,
            displayed_mix_opacity: "100".into(),
            mix_opacity_has_draft: false,
            mix_opacity_diagnostic: None,
            native_document,
            numeric_revisions: [revision; 2],
            numeric_diagnostic: None,
            numeric_drafts: [false; 2],
            numeric_input_diagnostics: [None, None],
            _subscriptions: subscriptions,
        }
    }

    pub(crate) fn discard_numeric_drafts(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let values = self.session.read(cx).numeric_document().map(|document| {
            document
                .inputs()
                .map(|inputs| (inputs, document.revision()))
        });
        let Some(values) = values else { return };
        match values {
            Ok(((augend, addend), revision)) => {
                self.numeric_revisions.fill(revision);
                self.numeric_drafts.fill(false);
                self.numeric_input_diagnostics.fill(None);
                self.numeric_diagnostic = None;
                for (editor, value) in
                    [(&self.augend_editor, augend), (&self.addend_editor, addend)]
                {
                    editor.update(cx, |editor, cx| {
                        editor.set_text(value.to_string(), window, cx)
                    });
                }
                if self.session.read(cx).current_numeric_result().is_none() {
                    let result = self
                        .session
                        .update(cx, |session, cx| session.request_numeric_evaluation(cx));
                    if let Err(error) = result {
                        self.numeric_diagnostic = Some(error.to_string());
                    }
                }
            }
            Err(error) => self.numeric_diagnostic = Some(error.to_string()),
        }
        cx.notify();
    }

    fn sync_numeric_fields(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(document) = self.session.read(cx).numeric_document() else {
            return;
        };
        let revision = document.revision();
        let (augend, addend) = match document.inputs() {
            Ok(inputs) => inputs,
            Err(error) => {
                self.numeric_diagnostic = Some(error.to_string());
                return;
            }
        };
        for (((editor, value), expected_revision), has_draft) in
            [(&self.augend_editor, augend), (&self.addend_editor, addend)]
                .into_iter()
                .zip(&mut self.numeric_revisions)
                .zip(&self.numeric_drafts)
        {
            if !has_draft && !editor.read(cx).is_focused(window) {
                *expected_revision = revision;
                let text = value.to_string();
                if editor.read(cx).text(cx) != text {
                    editor.update(cx, |editor, cx| editor.set_text(text, window, cx));
                }
            }
        }
    }

    pub(crate) fn set_mode(&mut self, mode: AddMode, cx: &mut Context<Self>) {
        let augend = self.augend_editor.read(cx).text(cx);
        let addend = self.addend_editor.read(cx).text(cx);
        let augend_y = self.augend_y_editor.read(cx).text(cx);
        let addend_y = self.addend_y_editor.read(cx).text(cx);
        self.session.update(cx, |session, cx| {
            session.set_mode_components(mode, &augend, &augend_y, &addend, &addend_y);
            cx.notify();
        });
    }

    fn display_mix_opacity(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.displayed_mix_opacity = self
            .session
            .read(cx)
            .mix_color_parameters()
            .opacity
            .to_string();
        self.mix_opacity_has_draft = false;
        self.mix_opacity_editor.update(cx, |editor, cx| {
            editor.set_text(self.displayed_mix_opacity.clone(), window, cx);
        });
    }

    fn set_mix_blend_mode(
        &mut self,
        blend_mode: graphics_core::resources::CpuBlendMode,
        cx: &mut Context<Self>,
    ) {
        let over = self.augend_editor.read(cx).text(cx);
        let under = self.addend_editor.read(cx).text(cx);
        self.session.update(cx, |session, cx| {
            let mut parameters = session.mix_color_parameters();
            parameters.blend_mode = blend_mode;
            session.set_mix_color_parameters(parameters, &over, &under);
            cx.notify();
        });
    }

    fn commit_mix_opacity(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.session.read(cx).mode() != AddMode::MixColor {
            return;
        }
        if !self.mix_opacity_has_draft {
            self.mix_opacity_diagnostic = None;
            self.display_mix_opacity(window, cx);
            cx.notify();
            return;
        }
        let expression = self.mix_opacity_editor.read(cx).text(cx);
        match graphics_core::numeric_input::evaluate_number_input(
            &expression,
            graphics_core::numeric_input::NumberInputConstraints {
                minimum: Some(0.0),
                maximum: Some(100.0),
                is_integer: false,
            },
        ) {
            Ok(opacity) => {
                self.mix_opacity_diagnostic = None;
                let over = self.augend_editor.read(cx).text(cx);
                let under = self.addend_editor.read(cx).text(cx);
                self.session.update(cx, |session, cx| {
                    let mut parameters = session.mix_color_parameters();
                    if parameters.opacity != opacity {
                        parameters.opacity = opacity;
                        session.set_mix_color_parameters(parameters, &over, &under);
                        cx.notify();
                    }
                });
                self.display_mix_opacity(window, cx);
            }
            Err(error) => self.mix_opacity_diagnostic = Some(error.to_string()),
        }
        cx.notify();
    }

    fn toggle_reverse(&mut self, cx: &mut Context<Self>) {
        let values = self.augend_editor.read(cx).text(cx);
        let keys = self.addend_editor.read(cx).text(cx);
        self.session.update(cx, |session, cx| {
            session.set_reverse(!session.reverse(), &values, &keys);
            cx.notify();
        });
    }
}

impl Render for GraphicsWorkspaceItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sync_numeric_fields(window, cx);
        let session = self.session.read(cx);
        let result = session.result_text();
        let local_diagnostic = self
            .numeric_input_diagnostics
            .iter()
            .flatten()
            .next()
            .cloned()
            .or_else(|| self.numeric_diagnostic.clone());
        let expression = if local_diagnostic.is_some() {
            format!("Last valid result: {}", session.result_text())
        } else {
            session.expression_text()
        };
        let mode = session.mode();
        let reverse = session.reverse();
        let native = session.numeric_document().is_some();
        let diagnostic = local_diagnostic.or_else(|| session.diagnostic().map(str::to_owned));
        let parameters = session.mix_color_parameters();
        if mode == AddMode::MixColor
            && self.displayed_mix_opacity != parameters.opacity.to_string()
            && !self.mix_opacity_has_draft
            && !self.mix_opacity_editor.read(cx).is_focused(window)
            && self.mix_opacity_diagnostic.is_none()
        {
            self.display_mix_opacity(window, cx);
        }
        let item = cx.entity().downgrade();
        let mix_menu = svg_blend_mode_menu(window, cx, move |mode, _, cx| {
            if let Some(item) = item.upgrade() {
                item.update(cx, |item, cx| item.set_mix_blend_mode(mode, cx));
            }
        });
        v_flex()
            .id("graphite-add")
            .track_focus(&self.focus_handle(cx))
            .size_full()
            .p_4()
            .gap_3()
            .bg(cx.theme().colors().editor_background)
            .child(Label::new(if mode == AddMode::MixColor {
                "Graphite Color Mix"
            } else if mode == AddMode::InvertColor {
                "Graphite Invert"
            } else if mode == AddMode::Sort {
                "Graphite Sort"
            } else if mode.is_multiply() {
                "Graphite Multiply"
            } else {
                "Graphite Add"
            }))
            .child(Label::new(if mode == AddMode::MixColor {
                "Edit four comma-separated gamma sRGB channels with straight alpha for Over and Under."
            } else if mode == AddMode::InvertColor {
                "Edit gamma sRGB red, green and blue with straight alpha. Invert preserves alpha."
            } else if mode == AddMode::Sort {
                "Edit the values to sort them. Leave the sort keys blank to sort by value."
            } else if mode.is_multiply() {
                "Edit the inputs to calculate their product."
            } else {
                "Edit the inputs to calculate their sum."
            }))
            .when(!native, |view| view.child(
                h_flex().flex_wrap().gap_2().children(
                    [
                        (AddMode::Scalar, "Scalar", "GRAPHITE-MODE-SCALAR"),
                        (AddMode::Lists, "Lists", "GRAPHITE-MODE-LISTS"),
                        (AddMode::Integer, "Integer", "GRAPHITE-MODE-INTEGER"),
                        (AddMode::Vector, "Vector + Vector", "GRAPHITE-MODE-VECTOR"),
                        (
                            AddMode::ScalarVector,
                            "Scalar + Vector",
                            "GRAPHITE-MODE-SCALAR-VECTOR",
                        ),
                        (
                            AddMode::VectorScalar,
                            "Vector + Scalar",
                            "GRAPHITE-MODE-VECTOR-SCALAR",
                        ),
                        (
                            AddMode::IntegerLists,
                            "Integer Lists",
                            "GRAPHITE-MODE-INTEGER-LISTS",
                        ),
                        (
                            AddMode::VectorLists,
                            "Vector Lists",
                            "GRAPHITE-MODE-VECTOR-LISTS",
                        ),
                        (
                            AddMode::ScalarVectorLists,
                            "Scalar + Vector Lists",
                            "GRAPHITE-MODE-SCALAR-VECTOR-LISTS",
                        ),
                        (
                            AddMode::VectorScalarLists,
                            "Vector + Scalar Lists",
                            "GRAPHITE-MODE-VECTOR-SCALAR-LISTS",
                        ),
                        (AddMode::MixColor, "Mix Color", "GRAPHITE-MODE-MIX-COLOR"),
                        (AddMode::Sort, "Sort", "GRAPHITE-MODE-SORT"),
                        (AddMode::InvertColor, "Invert Color", "GRAPHITE-MODE-INVERT-COLOR"),
                        (AddMode::MultiplyScalar, "Multiply Scalar", "GRAPHITE-MODE-MULTIPLY-SCALAR"),
                        (AddMode::MultiplyLists, "Multiply Lists", "GRAPHITE-MODE-MULTIPLY-LISTS"),
                        (AddMode::MultiplyInteger, "Multiply Integer", "GRAPHITE-MODE-MULTIPLY-INTEGER"),
                        (AddMode::MultiplyVector, "Multiply Vector × Vector", "GRAPHITE-MODE-MULTIPLY-VECTOR"),
                        (AddMode::MultiplyScalarVector, "Multiply Scalar × Vector", "GRAPHITE-MODE-MULTIPLY-SCALAR-VECTOR"),
                        (AddMode::MultiplyVectorScalar, "Multiply Vector × Scalar", "GRAPHITE-MODE-MULTIPLY-VECTOR-SCALAR"),
                        (AddMode::MultiplyIntegerLists, "Multiply Integer Lists", "GRAPHITE-MODE-MULTIPLY-INTEGER-LISTS"),
                        (AddMode::MultiplyVectorLists, "Multiply Vector Lists", "GRAPHITE-MODE-MULTIPLY-VECTOR-LISTS"),
                        (AddMode::MultiplyScalarVectorLists, "Multiply Scalar × Vector Lists", "GRAPHITE-MODE-MULTIPLY-SCALAR-VECTOR-LISTS"),
                        (AddMode::MultiplyVectorScalarLists, "Multiply Vector × Scalar Lists", "GRAPHITE-MODE-MULTIPLY-VECTOR-SCALAR-LISTS"),
                        (AddMode::MultiplyAffine, "Multiply Affine", "GRAPHITE-MODE-MULTIPLY-AFFINE"),
                        (AddMode::MultiplyAffineLists, "Multiply Affine Lists", "GRAPHITE-MODE-MULTIPLY-AFFINE-LISTS"),

                    ]
                    .into_iter()
                    .map(|(candidate, label, selector)| {
                        div().debug_selector(move || selector.into()).child(
                            Button::new(selector, label)
                                .style(if mode == candidate {
                                    ButtonStyle::Tinted(TintColor::Accent)
                                } else {
                                    ButtonStyle::Subtle
                                })
                                .on_click(
                                    cx.listener(move |this, _, _, cx| this.set_mode(candidate, cx)),
                                ),
                        )
                    }),
                ),
            ))
            .when(native, |view| view.child(
                h_flex().gap_2()
                    .child(div().debug_selector(|| "GRAPHITE-NUMERIC-REFRESH".into()).child(
                        Button::new("graphite-numeric-refresh", "Evaluate")
                            .on_click(cx.listener(|item, _, _, cx| {
                                let result = item.session.update(cx, |session, cx| session.request_numeric_evaluation(cx));
                                if let Err(error) = result { item.numeric_diagnostic = Some(error.to_string()); }
                                cx.notify();
                            }))))
                    .child(div().debug_selector(|| "GRAPHITE-NUMERIC-DISCARD-DRAFTS".into()).child(
                        Button::new("graphite-numeric-discard-drafts", "Discard input drafts")
                            .on_click(cx.listener(|item, _, window, cx| item.discard_numeric_drafts(window, cx)))))
                    .child(div().debug_selector(|| "GRAPHITE-NUMERIC-CANCEL".into()).child(
                        Button::new("graphite-numeric-cancel", "Cancel evaluation")
                            .on_click(cx.listener(|item, _, _, cx| {
                                item.session.update(cx, |session, cx| session.cancel_numeric_evaluation(cx));
                                cx.notify();
                            }))))
            ))
            .when(mode == AddMode::MixColor, |view| {
                view.child(h_flex().gap_2()
                    .child(div().debug_selector(|| "GRAPHITE-COLOR-MIX-BLEND-MODE".into())
                        .child(DropdownMenu::new("graphite-color-mix-mode", parameters.blend_mode.to_string(), mix_menu)))
                    .child(Label::new("Opacity (%)"))
                    .child(div().id("graphite-color-mix-opacity").debug_selector(|| "GRAPHITE-COLOR-MIX-OPACITY".into())
                        .aria_label("Color Mix opacity percentage")
                        .w_24().child(self.mix_opacity_editor.clone())
                        .on_action(cx.listener(|item, _: &menu::Confirm, window, cx| {
                            item.commit_mix_opacity(window, cx);
                            item.augend_editor.focus_handle(cx).focus(window, cx);
                            cx.stop_propagation();
                        }))
                        .on_action(cx.listener(|item, _: &menu::Cancel, window, cx| {
                            item.mix_opacity_diagnostic = None;
                            item.display_mix_opacity(window, cx);
                            item.augend_editor.focus_handle(cx).focus(window, cx);
                            cx.stop_propagation();
                            cx.notify();
                        }))))
                    .when_some(self.mix_opacity_diagnostic.clone(), |view, diagnostic| {
                        view.child(div().id("graphite-color-mix-opacity-error").role(gpui::Role::Alert)
                            .debug_selector(|| "GRAPHITE-COLOR-MIX-OPACITY-ERROR".into())
                            .child(Label::new(diagnostic).color(Color::Error)))
                    })
            })
            .when(mode == AddMode::Sort, |view| {
                view.child(Label::new(
                    "Separate numbers with commas. Leave values blank for an empty list.",
                ))
                .child(
                    div()
                        .debug_selector(|| "GRAPHITE-SORT-REVERSE".into())
                        .child(
                            Button::new(
                                "graphite-sort-reverse",
                                if reverse {
                                    "Reverse: On"
                                } else {
                                    "Reverse: Off"
                                },
                            )
                            .style(if reverse {
                                ButtonStyle::Tinted(TintColor::Accent)
                            } else {
                                ButtonStyle::Subtle
                            })
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_reverse(cx))),
                        ),
                )
            })
            .when(mode.is_list() && !mode.is_affine(), |view| {
                view.child(Label::new(
                    "Separate numbers with commas. Leave an input blank for an empty list.",
                ))
            })
            .when(mode.is_affine(), |view| {
                view.child(Label::new(if mode.is_list() {
                    "Enter six comma-separated components per transform; separate transforms with semicolons. Leave an input blank for an empty list."
                } else {
                    "Enter six comma-separated components per transform."
                }))
                .child(Label::new("Component order: m11, m12, m21, m22, tx, ty (column-major)."))
            })
            .when(
                matches!(mode, AddMode::Integer | AddMode::IntegerLists | AddMode::MultiplyInteger | AddMode::MultiplyIntegerLists),
                |view| {
                    view.child(Label::new(
                        "Enter whole numbers in the 64-bit integer range.",
                    ))
                },
            )
            .when(
                mode.is_list() && (mode.augend_is_vector() || mode.addend_is_vector()),
                |view| {
                    view.child(Label::new(
                        "Pair x and y components in order, with the same number in each column.",
                    ))
                },
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new(if mode == AddMode::MixColor {
                        "Over (red, green, blue, alpha)"
                    } else if mode == AddMode::InvertColor {
                        "red"
                    } else if mode == AddMode::Sort {
                        "values"
                    } else if mode.is_multiply() {
                        if mode.augend_is_vector() { "multiplier.x" } else { "multiplier" }
                    } else if mode.augend_is_vector() {
                        "augend.x"
                    } else {
                        "augend"
                    }))
                    .child(
                        div()
                            .id("augend")
                            .aria_label(if mode == AddMode::MixColor {
                                "Over RGBA"
                            } else if mode == AddMode::InvertColor {
                                "red"
                            } else if mode == AddMode::Sort {
                                "values"
                            } else if mode.is_multiply() {
                                "multiplier"
                            } else {
                                "augend"
                            })
                            .child(self.augend_editor.clone()),
                    ),
            )
            .when(mode.augend_is_vector() || mode == AddMode::InvertColor, |view| {
                view.child(
                    v_flex().gap_1().child(Label::new(if mode == AddMode::InvertColor { "green" } else if mode.is_multiply() { "multiplier.y" } else { "augend.y" })).child(
                        div()
                            .id("augend-y")
                            .aria_label(if mode == AddMode::InvertColor { "green" } else if mode.is_multiply() { "multiplier.y" } else { "augend.y" })
                            .child(self.augend_y_editor.clone()),
                    ),
                )
            })
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new(if mode == AddMode::MixColor {
                        "Under (red, green, blue, alpha)"
                    } else if mode == AddMode::InvertColor {
                        "blue"
                    } else if mode == AddMode::Sort {
                        "sort keys (optional)"
                    } else if mode.is_multiply() {
                        if mode.addend_is_vector() { "multiplicand.x" } else { "multiplicand" }
                    } else if mode.addend_is_vector() {
                        "addend.x"
                    } else {
                        "addend"
                    }))
                    .child(
                        div()
                            .id("addend")
                            .aria_label(if mode == AddMode::MixColor {
                                "Under RGBA"
                            } else if mode == AddMode::InvertColor {
                                "blue"
                            } else if mode == AddMode::Sort {
                                "sort keys"
                            } else if mode.is_multiply() {
                                "multiplicand"
                            } else {
                                "addend"
                            })
                            .child(self.addend_editor.clone()),
                    ),
            )
            .when(mode.addend_is_vector() || mode == AddMode::InvertColor, |view| {
                view.child(
                    v_flex().gap_1().child(Label::new(if mode == AddMode::InvertColor { "alpha" } else if mode.is_multiply() { "multiplicand.y" } else { "addend.y" })).child(
                        div()
                            .id("addend-y")
                            .aria_label(if mode == AddMode::InvertColor { "alpha" } else if mode.is_multiply() { "multiplicand.y" } else { "addend.y" })
                            .child(self.addend_y_editor.clone()),
                    ),
                )
            })
            .child(
                div().debug_selector(|| "GRAPHITE-RESULT".into()).child(
                    div()
                        .debug_selector(move || format!("GRAPHITE-RESULT-{result}"))
                        .child(Label::new(expression)),
                ),
            )
            .when_some(diagnostic, |view, diagnostic| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-INPUT-ERROR".into())
                        .child(
                            Label::new(format!("{diagnostic} Showing the last valid result."))
                                .color(Color::Error),
                        ),
                )
            })
    }
}

impl Focusable for GraphicsWorkspaceItem {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.augend_editor.focus_handle(cx)
    }
}

impl EventEmitter<()> for GraphicsWorkspaceItem {}

impl Item for GraphicsWorkspaceItem {
    type Event = ();

    fn can_split(&self) -> bool {
        self.native_document
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Task<Option<Entity<Self>>> {
        let session = self.session.clone();
        let values = session
            .read(cx)
            .numeric_document()
            .map(|document| document.inputs());
        match values {
            Some(Ok((augend, addend))) => gpui::Task::ready(Some(cx.new(|cx| {
                Self::with_session(session, augend.to_string(), addend.to_string(), window, cx)
            }))),
            Some(Err(error)) => {
                log::error!("Cannot split invalid native Graphite document: {error}");
                gpui::Task::ready(None)
            }
            None => gpui::Task::ready(None),
        }
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        if self.session.read(cx).mode() == AddMode::MixColor {
            "Graphite Color Mix".into()
        } else if self.session.read(cx).mode() == AddMode::InvertColor {
            "Graphite Invert".into()
        } else if self.session.read(cx).mode() == AddMode::Sort {
            "Graphite Sort".into()
        } else if self.session.read(cx).mode().is_multiply() {
            "Graphite Multiply".into()
        } else {
            "Graphite Add".into()
        }
    }
}

pub fn open_numeric_document(
    document: graphics_core::document::NumericAddDocument,
    window: &mut Window,
    cx: &mut App,
) -> Result<Entity<GraphicsWorkspaceItem>, graphics_core::document::NumericDocumentError> {
    let (augend, addend) = document.inputs()?;
    let session_value = GraphicsSession::from_numeric_document(document)?;
    let session = cx.new(|_| session_value);
    session.update(cx, |session, cx| session.request_numeric_evaluation(cx))?;
    Ok(cx.new(|cx| {
        GraphicsWorkspaceItem::with_session(
            session,
            augend.to_string(),
            addend.to_string(),
            window,
            cx,
        )
    }))
}

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        gpui::KeyBinding::new("tab", FocusNextArtboardControl, Some("GraphiteArtboard")),
        gpui::KeyBinding::new(
            "shift-tab",
            FocusPreviousArtboardControl,
            Some("GraphiteArtboard"),
        ),
    ]);
}

pub fn init(cx: &mut App) {
    bind_keys(cx);
    cx.observe_new(|workspace: &mut Workspace, _, _cx| {
        workspace.register_action(|workspace, _: &OpenGraphite, window, cx| {
            let item = cx.new(|cx| GraphicsWorkspaceItem::new(window, cx));
            workspace.active_pane().update(cx, |pane, cx| {
                pane.add_item(Box::new(item), true, true, None, window, cx);
            });
        });
    })
    .detach();
}

actions!(
    graphite,
    [
        /// Shows the original imported image in the focused Graphite item.
        ShowOriginalImage,
        /// Inverts the focused imported image with the original Graphite node.
        InvertImage,
        /// Makes the focused imported image opaque with the original Graphite node.
        MakeImageOpaque,
        /// Cancels pending work in the focused imported-image session.
        CancelImageLoad
    ]
);

const BLACK_AND_WHITE_TEMPLATE_PARAMETERS: graphics_core::resources::CpuBlackAndWhiteParameters =
    graphics_core::resources::CpuBlackAndWhiteParameters {
        use_tint: false,
        tint_gamma_srgb: [225. / 255., 211. / 255., 179. / 255., 1.],
        reds: 40.,
        yellows: 60.,
        greens: 40.,
        cyans: 60.,
        blues: 20.,
        magentas: 80.,
    };

#[derive(Clone, Copy)]
enum BlackAndWhiteField {
    Tint,
    Reds,
    Yellows,
    Greens,
    Cyans,
    Blues,
    Magentas,
}

impl BlackAndWhiteField {
    const ALL: [Self; 7] = [
        Self::Tint,
        Self::Reds,
        Self::Yellows,
        Self::Greens,
        Self::Cyans,
        Self::Blues,
        Self::Magentas,
    ];
    fn index(self) -> usize {
        match self {
            Self::Tint => 0,
            Self::Reds => 1,
            Self::Yellows => 2,
            Self::Greens => 3,
            Self::Cyans => 4,
            Self::Blues => 5,
            Self::Magentas => 6,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Tint => "Tint (gamma-sRGB RGBA)",
            Self::Reds => "Reds (%)",
            Self::Yellows => "Yellows (%)",
            Self::Greens => "Greens (%)",
            Self::Cyans => "Cyans (%)",
            Self::Blues => "Blues (%)",
            Self::Magentas => "Magentas (%)",
        }
    }
    fn selector(self) -> &'static str {
        match self {
            Self::Tint => "GRAPHITE-BLACK-WHITE-TINT",
            Self::Reds => "GRAPHITE-BLACK-WHITE-REDS",
            Self::Yellows => "GRAPHITE-BLACK-WHITE-YELLOWS",
            Self::Greens => "GRAPHITE-BLACK-WHITE-GREENS",
            Self::Cyans => "GRAPHITE-BLACK-WHITE-CYANS",
            Self::Blues => "GRAPHITE-BLACK-WHITE-BLUES",
            Self::Magentas => "GRAPHITE-BLACK-WHITE-MAGENTAS",
        }
    }
    fn text(self, parameters: graphics_core::resources::CpuBlackAndWhiteParameters) -> String {
        match self {
            Self::Tint => parameters
                .tint_gamma_srgb
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            Self::Reds => parameters.reds.to_string(),
            Self::Yellows => parameters.yellows.to_string(),
            Self::Greens => parameters.greens.to_string(),
            Self::Cyans => parameters.cyans.to_string(),
            Self::Blues => parameters.blues.to_string(),
            Self::Magentas => parameters.magentas.to_string(),
        }
    }
}

const BRIGHTNESS_CONTRAST_TEMPLATE_PARAMETERS:
    graphics_core::resources::CpuBrightnessContrastParameters =
    graphics_core::resources::CpuBrightnessContrastParameters {
        brightness: 0.,
        contrast: 0.,
        use_classic: false,
        classic_pivot: 127.,
    };

#[derive(Clone, Copy)]
enum BrightnessContrastField {
    Brightness,
    Contrast,
    ClassicPivot,
}

impl BrightnessContrastField {
    const ALL: [Self; 3] = [Self::Brightness, Self::Contrast, Self::ClassicPivot];

    fn index(self) -> usize {
        match self {
            Self::Brightness => 0,
            Self::Contrast => 1,
            Self::ClassicPivot => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Brightness => "Brightness (%)",
            Self::Contrast => "Contrast (%)",
            Self::ClassicPivot => "Classic pivot",
        }
    }

    fn selector(self) -> &'static str {
        match self {
            Self::Brightness => "GRAPHITE-BRIGHTNESS-CONTRAST-BRIGHTNESS",
            Self::Contrast => "GRAPHITE-BRIGHTNESS-CONTRAST-CONTRAST",
            Self::ClassicPivot => "GRAPHITE-BRIGHTNESS-CONTRAST-CLASSIC-PIVOT",
        }
    }

    fn value(self, parameters: graphics_core::resources::CpuBrightnessContrastParameters) -> f64 {
        match self {
            Self::Brightness => parameters.brightness,
            Self::Contrast => parameters.contrast,
            Self::ClassicPivot => parameters.classic_pivot,
        }
    }

    fn visible(
        self,
        parameters: graphics_core::resources::CpuBrightnessContrastParameters,
    ) -> bool {
        !matches!(self, Self::ClassicPivot) || parameters.use_classic
    }

    fn constraints(
        self,
        parameters: graphics_core::resources::CpuBrightnessContrastParameters,
    ) -> graphics_core::numeric_input::NumberInputConstraints {
        let (minimum, maximum) = match (self, parameters.use_classic) {
            (Self::Brightness, false) => (-150., 150.),
            (Self::Brightness | Self::Contrast, true) => (-100., 100.),
            (Self::Contrast, false) => (-50., 100.),
            (Self::ClassicPivot, _) => (0., 255.),
        };
        graphics_core::numeric_input::NumberInputConstraints {
            minimum: Some(minimum),
            maximum: Some(maximum),
            is_integer: false,
        }
    }
}

struct PendingItemScenePreview {
    generation: u64,
    _task: gpui::Task<()>,
}

pub struct ImportedImageItem {
    session: Entity<crate::session::GraphiteImageSession>,
    presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
    max_preview_bytes: usize,
    preview_diagnostic: Option<String>,
    scene_acceptance_diagnostic: Option<String>,
    scene_configuration: Option<graphics_core::render::CpuSvgSceneConfiguration>,
    scene_request_generation: u64,
    pending_scene_preview: Option<PendingItemScenePreview>,
    pub(crate) black_and_white_editors: [Entity<Editor>; 7],
    black_and_white_parameters: graphics_core::resources::CpuBlackAndWhiteParameters,
    black_and_white_displayed: [String; 7],
    black_and_white_drafts: [bool; 7],
    black_and_white_diagnostics: [Option<String>; 7],
    pub(crate) brightness_contrast_editors: [Entity<Editor>; 3],
    brightness_contrast_parameters: graphics_core::resources::CpuBrightnessContrastParameters,
    brightness_contrast_displayed: [String; 3],
    brightness_contrast_drafts: [bool; 3],
    brightness_contrast_diagnostics: [Option<String>; 3],
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl ImportedImageItem {
    pub fn from_session(
        session: Entity<crate::session::GraphiteImageSession>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        Self::from_shared(session, presentation, max_preview_bytes, window, cx)
    }

    /// Keeps the host-supplied target fixed instead of deriving a camera from window or split bounds.
    pub fn from_session_with_scene_configuration(
        session: Entity<crate::session::GraphiteImageSession>,
        max_preview_bytes: usize,
        configuration: graphics_core::render::CpuSvgSceneConfiguration,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut item = Self::from_session(session, max_preview_bytes, window, cx);
        item.scene_configuration = Some(configuration);
        item
    }

    fn from_shared(
        session: Entity<crate::session::GraphiteImageSession>,
        presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let parameters = match session.read(cx).operation() {
            Some(graphics_core::resources::CpuImageEdit::BlackAndWhite(parameters)) => parameters,
            _ => BLACK_AND_WHITE_TEMPLATE_PARAMETERS,
        };
        let displayed = BlackAndWhiteField::ALL.map(|field| field.text(parameters));
        let editors = BlackAndWhiteField::ALL.map(|field| {
            cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_text(field.text(parameters), window, cx);
                editor.set_placeholder_text(field.label(), window, cx);
                editor.set_read_only(
                    matches!(field, BlackAndWhiteField::Tint) && !parameters.use_tint,
                );
                editor
            })
        });
        let brightness_contrast_parameters = match session.read(cx).operation() {
            Some(graphics_core::resources::CpuImageEdit::BrightnessContrast(parameters)) => {
                parameters
            }
            _ => BRIGHTNESS_CONTRAST_TEMPLATE_PARAMETERS,
        };
        let brightness_contrast_displayed = BrightnessContrastField::ALL
            .map(|field| field.value(brightness_contrast_parameters).to_string());
        let brightness_contrast_editors = BrightnessContrastField::ALL.map(|field| {
            cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_text(
                    field.value(brightness_contrast_parameters).to_string(),
                    window,
                    cx,
                );
                editor.set_placeholder_text(field.label(), window, cx);
                editor
            })
        });
        let mut subscriptions = vec![
            cx.observe(&session, |_, _, cx| cx.notify()),
            cx.observe(&presentation, |_, _, cx| cx.notify()),
        ];
        for (field, editor) in BlackAndWhiteField::ALL.into_iter().zip(&editors) {
            subscriptions.push(cx.subscribe_in(
                editor,
                window,
                move |item, editor, event, window, cx| match event {
                    editor::EditorEvent::BufferEdited => {
                        item.black_and_white_drafts[field.index()] = editor.read(cx).text(cx)
                            != item.black_and_white_displayed[field.index()];
                        cx.notify();
                    }
                    editor::EditorEvent::Blurred => item.commit_black_and_white(field, window, cx),
                    _ => {}
                },
            ));
        }
        for (field, editor) in BrightnessContrastField::ALL
            .into_iter()
            .zip(&brightness_contrast_editors)
        {
            subscriptions.push(cx.subscribe_in(
                editor,
                window,
                move |item, editor, event, window, cx| match event {
                    editor::EditorEvent::BufferEdited => {
                        item.brightness_contrast_drafts[field.index()] = editor.read(cx).text(cx)
                            != item.brightness_contrast_displayed[field.index()];
                        cx.notify();
                    }
                    editor::EditorEvent::Blurred => {
                        item.commit_brightness_contrast(field, window, cx)
                    }
                    _ => {}
                },
            ));
        }
        Self {
            session,
            presentation,
            max_preview_bytes,
            preview_diagnostic: None,
            scene_acceptance_diagnostic: None,
            scene_configuration: None,
            scene_request_generation: 0,
            pending_scene_preview: None,
            black_and_white_editors: editors,
            black_and_white_parameters: parameters,
            black_and_white_displayed: displayed,
            black_and_white_drafts: [false; 7],
            black_and_white_diagnostics: std::array::from_fn(|_| None),
            brightness_contrast_editors,
            brightness_contrast_parameters,
            brightness_contrast_displayed,
            brightness_contrast_drafts: [false; 3],
            brightness_contrast_diagnostics: std::array::from_fn(|_| None),
            focus_handle: cx.focus_handle(),
            _subscriptions: subscriptions,
        }
    }

    pub fn session(&self) -> &Entity<crate::session::GraphiteImageSession> {
        &self.session
    }

    pub fn presentation(&self) -> &Entity<crate::image_preview::CpuPreviewPresentation> {
        &self.presentation
    }

    pub fn accept_scene_preview(
        &mut self,
        preview: crate::session::PreparedImageScenePreview,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Result<(), crate::image_preview::CpuPreviewError> {
        let admission = if !preview.is_exact_target() || preview.image.frame_count() != 1 {
            Err(crate::image_preview::CpuPreviewError::InvalidPixelLayout)
        } else {
            match preview.image.as_bytes(0) {
                Some(pixels) if pixels.len() > self.max_preview_bytes => {
                    Err(crate::image_preview::CpuPreviewError::PreviewBudgetExceeded)
                }
                Some(_) => Ok(()),
                None => Err(crate::image_preview::CpuPreviewError::InvalidPixelLayout),
            }
        };
        let result = admission.and_then(|()| {
            let presentation = self.presentation.clone();
            self.session.update(cx, |session, cx| {
                presentation.update(cx, |presentation, cx| {
                    presentation.accept_scene_preview(session, preview, window, cx)
                })
            })
        });
        self.scene_acceptance_diagnostic = result.as_ref().err().map(ToString::to_string);
        cx.notify();
        result
    }

    fn request_scene_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.pending_scene_preview.is_some() {
            return;
        }
        let Some(configuration) = self.scene_configuration else {
            self.scene_acceptance_diagnostic = Some("A scene target has not been supplied".into());
            cx.notify();
            return;
        };
        let Some(generation) = self.scene_request_generation.checked_add(1) else {
            self.scene_acceptance_diagnostic = Some("Scene request generation is exhausted".into());
            cx.notify();
            return;
        };
        let max_preview_bytes = self.max_preview_bytes;
        let preparation = match self.session.update(cx, |session, cx| {
            session.prepare_scene_preview_at_target(configuration, max_preview_bytes, cx)
        }) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.scene_acceptance_diagnostic = Some(error);
                cx.notify();
                return;
            }
        };
        self.scene_request_generation = generation;
        self.scene_acceptance_diagnostic = None;
        let task = cx.spawn_in(window, async move |item, cx| {
            let outcome = preparation.await;
            if let Err(error) = item.update_in(cx, |item, window, cx| {
                if !item
                    .pending_scene_preview
                    .as_ref()
                    .is_some_and(|pending| pending.generation == generation)
                {
                    return;
                }
                item.pending_scene_preview = None;
                match outcome {
                    Ok(preview) => {
                        if let Err(error) = item.accept_scene_preview(preview, window, cx) {
                            item.scene_acceptance_diagnostic = Some(error.to_string());
                        }
                    }
                    Err(error) => item.scene_acceptance_diagnostic = Some(error),
                }
                cx.notify();
            }) {
                log::debug!("Scene preview Item or window was released: {error}");
            }
        });
        self.pending_scene_preview = Some(PendingItemScenePreview {
            generation,
            _task: task,
        });
        cx.notify();
    }

    pub fn preview_diagnostic(&self) -> Option<&str> {
        self.preview_diagnostic.as_deref()
    }

    fn select_operation(
        &mut self,
        operation: Option<graphics_core::resources::CpuImageEdit>,
        cx: &mut Context<Self>,
    ) {
        self.session
            .update(cx, |session, cx| session.set_operation(operation, cx));
    }

    pub(crate) fn black_and_white_diagnostic(&self) -> Option<&str> {
        self.black_and_white_diagnostics
            .iter()
            .find_map(|diagnostic| diagnostic.as_deref())
    }

    fn current_black_and_white_parameters(
        &self,
        cx: &App,
    ) -> graphics_core::resources::CpuBlackAndWhiteParameters {
        match self.session.read(cx).operation() {
            Some(graphics_core::resources::CpuImageEdit::BlackAndWhite(parameters)) => parameters,
            _ => self.black_and_white_parameters,
        }
    }

    fn display_black_and_white_field(
        &mut self,
        field: BlackAndWhiteField,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text = field.text(self.current_black_and_white_parameters(cx));
        self.black_and_white_displayed[field.index()] = text.clone();
        self.black_and_white_drafts[field.index()] = false;
        self.black_and_white_editors[field.index()]
            .update(cx, |editor, cx| editor.set_text(text, window, cx));
    }

    fn commit_black_and_white(
        &mut self,
        field: BlackAndWhiteField,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        use graphics_core::resources::CpuImageEdit;
        let Some(CpuImageEdit::BlackAndWhite(mut parameters)) = self.session.read(cx).operation()
        else {
            return;
        };
        if matches!(field, BlackAndWhiteField::Tint) && !parameters.use_tint {
            return;
        }
        if !self.black_and_white_drafts[field.index()] {
            return;
        }
        let text = self.black_and_white_editors[field.index()]
            .read(cx)
            .text(cx);
        let parsed = match field {
            BlackAndWhiteField::Tint => crate::session::parse_color(&text, "Tint")
                .map(|tint| parameters.tint_gamma_srgb = tint),
            _ => graphics_core::numeric_input::evaluate_number_input(
                &text,
                graphics_core::numeric_input::NumberInputConstraints {
                    minimum: Some(-200.),
                    maximum: Some(300.),
                    is_integer: false,
                },
            )
            .map(|value| match field {
                BlackAndWhiteField::Reds => parameters.reds = value,
                BlackAndWhiteField::Yellows => parameters.yellows = value,
                BlackAndWhiteField::Greens => parameters.greens = value,
                BlackAndWhiteField::Cyans => parameters.cyans = value,
                BlackAndWhiteField::Blues => parameters.blues = value,
                BlackAndWhiteField::Magentas => parameters.magentas = value,
                BlackAndWhiteField::Tint => {}
            })
            .map_err(|error| error.to_string()),
        };
        match parsed {
            Ok(()) => {
                self.black_and_white_parameters = parameters;
                self.black_and_white_diagnostics[field.index()] = None;
                self.select_operation(Some(CpuImageEdit::BlackAndWhite(parameters)), cx);
                self.display_black_and_white_field(field, window, cx);
            }
            Err(error) => self.black_and_white_diagnostics[field.index()] = Some(error),
        }
        cx.notify();
    }

    fn black_and_white_controls(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let parameters = self.current_black_and_white_parameters(cx);
        self.black_and_white_parameters = parameters;
        for field in BlackAndWhiteField::ALL {
            let editor = &self.black_and_white_editors[field.index()];
            if !self.black_and_white_drafts[field.index()]
                && self.black_and_white_diagnostics[field.index()].is_none()
                && !editor.read(cx).is_focused(window)
                && editor.read(cx).text(cx) != field.text(parameters)
            {
                self.display_black_and_white_field(field, window, cx);
            }
        }
        self.black_and_white_editors[0]
            .update(cx, |editor, _| editor.set_read_only(!parameters.use_tint));
        v_flex()
            .gap_1()
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-BLACK-WHITE-USE-TINT".into())
                    .child(
                        Button::new(
                            "graphite-black-white-use-tint",
                            if parameters.use_tint {
                                "Tint: On"
                            } else {
                                "Tint: Off"
                            },
                        )
                        .toggle_state(parameters.use_tint)
                        .on_click(cx.listener(|item, _, _, cx| {
                            let mut parameters = item.current_black_and_white_parameters(cx);
                            parameters.use_tint = !parameters.use_tint;
                            item.black_and_white_parameters = parameters;
                            item.select_operation(
                                Some(graphics_core::resources::CpuImageEdit::BlackAndWhite(
                                    parameters,
                                )),
                                cx,
                            );
                        })),
                    ),
            )
            .child(Label::new(
                "Tint channels: red, green, blue, alpha (gamma-sRGB).",
            ))
            .children(BlackAndWhiteField::ALL.into_iter().map(|field| {
                h_flex()
                    .gap_2()
                    .child(Label::new(field.label()))
                    .child(
                        div()
                            .id(field.selector())
                            .debug_selector(move || field.selector().into())
                            .aria_label(field.label())
                            .w_64()
                            .child(self.black_and_white_editors[field.index()].clone())
                            .on_action(cx.listener(move |item, _: &menu::Confirm, window, cx| {
                                item.commit_black_and_white(field, window, cx);
                                item.focus_handle.focus(window, cx);
                                cx.stop_propagation();
                            }))
                            .on_action(cx.listener(move |item, _: &menu::Cancel, window, cx| {
                                item.black_and_white_diagnostics[field.index()] = None;
                                item.display_black_and_white_field(field, window, cx);
                                item.focus_handle.focus(window, cx);
                                cx.stop_propagation();
                                cx.notify();
                            })),
                    )
                    .child(
                        div()
                            .debug_selector(move || format!("{}-RESET", field.selector()))
                            .child(
                                Button::new(("graphite-black-white-reset", field.index()), "Reset")
                                    .disabled(
                                        matches!(field, BlackAndWhiteField::Tint)
                                            && !parameters.use_tint,
                                    )
                                    .on_click(cx.listener(move |item, _, window, cx| {
                                        let text = field.text(BLACK_AND_WHITE_TEMPLATE_PARAMETERS);
                                        item.black_and_white_drafts[field.index()] =
                                            text != item.black_and_white_displayed[field.index()];
                                        item.black_and_white_editors[field.index()]
                                            .update(cx, |editor, cx| {
                                                editor.set_text(text, window, cx)
                                            });
                                        item.commit_black_and_white(field, window, cx);
                                    })),
                            ),
                    )
            }))
            .when_some(
                self.black_and_white_diagnostic().map(str::to_owned),
                |view, diagnostic| {
                    view.child(
                        div()
                            .id("graphite-black-white-error")
                            .role(gpui::Role::Alert)
                            .debug_selector(|| "GRAPHITE-BLACK-WHITE-ERROR".into())
                            .child(Label::new(diagnostic).color(Color::Error)),
                    )
                },
            )
    }

    fn current_brightness_contrast_parameters(
        &self,
        cx: &App,
    ) -> graphics_core::resources::CpuBrightnessContrastParameters {
        match self.session.read(cx).operation() {
            Some(graphics_core::resources::CpuImageEdit::BrightnessContrast(parameters)) => {
                parameters
            }
            _ => self.brightness_contrast_parameters,
        }
    }

    pub(crate) fn brightness_contrast_diagnostic(&self, cx: &App) -> Option<&str> {
        let parameters = self.current_brightness_contrast_parameters(cx);
        BrightnessContrastField::ALL
            .into_iter()
            .filter(|field| field.visible(parameters))
            .find_map(|field| self.brightness_contrast_diagnostics[field.index()].as_deref())
    }

    fn brightness_contrast_has_draft(&self, cx: &App) -> bool {
        let parameters = self.current_brightness_contrast_parameters(cx);
        BrightnessContrastField::ALL.into_iter().any(|field| {
            field.visible(parameters) && self.brightness_contrast_drafts[field.index()]
        })
    }

    fn display_brightness_contrast_field(
        &mut self,
        field: BrightnessContrastField,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text = field
            .value(self.current_brightness_contrast_parameters(cx))
            .to_string();
        self.brightness_contrast_displayed[field.index()] = text.clone();
        self.brightness_contrast_drafts[field.index()] = false;
        self.brightness_contrast_editors[field.index()]
            .update(cx, |editor, cx| editor.set_text(text, window, cx));
    }

    fn commit_brightness_contrast(
        &mut self,
        field: BrightnessContrastField,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        use graphics_core::resources::CpuImageEdit;
        let Some(CpuImageEdit::BrightnessContrast(mut parameters)) =
            self.session.read(cx).operation()
        else {
            return;
        };
        if !field.visible(parameters) || !self.brightness_contrast_drafts[field.index()] {
            return;
        }
        let text = self.brightness_contrast_editors[field.index()]
            .read(cx)
            .text(cx);
        match graphics_core::numeric_input::evaluate_number_input(
            &text,
            field.constraints(parameters),
        ) {
            Ok(value) => {
                match field {
                    BrightnessContrastField::Brightness => parameters.brightness = value,
                    BrightnessContrastField::Contrast => parameters.contrast = value,
                    BrightnessContrastField::ClassicPivot => parameters.classic_pivot = value,
                }
                self.brightness_contrast_parameters = parameters;
                self.brightness_contrast_diagnostics[field.index()] = None;
                self.select_operation(Some(CpuImageEdit::BrightnessContrast(parameters)), cx);
                self.display_brightness_contrast_field(field, window, cx);
            }
            Err(error) => {
                self.brightness_contrast_diagnostics[field.index()] = Some(error.to_string())
            }
        }
        cx.notify();
    }

    fn brightness_contrast_controls(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let parameters = self.current_brightness_contrast_parameters(cx);
        self.brightness_contrast_parameters = parameters;
        for field in BrightnessContrastField::ALL {
            let editor = &self.brightness_contrast_editors[field.index()];
            if !self.brightness_contrast_drafts[field.index()]
                && self.brightness_contrast_diagnostics[field.index()].is_none()
                && !editor.read(cx).is_focused(window)
                && editor.read(cx).text(cx) != field.value(parameters).to_string()
            {
                self.display_brightness_contrast_field(field, window, cx);
            }
        }
        v_flex()
            .gap_1()
            .child(
                div()
                    .debug_selector(|| "GRAPHITE-BRIGHTNESS-CONTRAST-USE-CLASSIC".into())
                    .child(
                        Button::new("graphite-brightness-contrast-use-classic", "Use Classic")
                            .toggle_state(parameters.use_classic)
                            .on_click(cx.listener(|item, _, window, cx| {
                                let mut parameters =
                                    item.current_brightness_contrast_parameters(cx);
                                parameters.use_classic = !parameters.use_classic;
                                item.brightness_contrast_parameters = parameters;
                                item.select_operation(
                                    Some(
                                        graphics_core::resources::CpuImageEdit::BrightnessContrast(
                                            parameters,
                                        ),
                                    ),
                                    cx,
                                );
                                if !parameters.use_classic
                                    && item.brightness_contrast_editors[2]
                                        .read(cx)
                                        .is_focused(window)
                                {
                                    item.focus_handle.focus(window, cx);
                                }
                            })),
                    ),
            )
            .children(
                BrightnessContrastField::ALL
                    .into_iter()
                    .filter(|field| field.visible(parameters))
                    .map(|field| {
                        h_flex()
                            .gap_2()
                            .child(Label::new(field.label()))
                            .child(
                                div()
                                    .id(field.selector())
                                    .debug_selector(move || field.selector().into())
                                    .aria_label(field.label())
                                    .w_64()
                                    .child(self.brightness_contrast_editors[field.index()].clone())
                                    .on_action(cx.listener(
                                        move |item, _: &menu::Confirm, window, cx| {
                                            item.commit_brightness_contrast(field, window, cx);
                                            item.focus_handle.focus(window, cx);
                                            cx.stop_propagation();
                                        },
                                    ))
                                    .on_action(cx.listener(
                                        move |item, _: &menu::Cancel, window, cx| {
                                            item.brightness_contrast_diagnostics[field.index()] =
                                                None;
                                            item.display_brightness_contrast_field(
                                                field, window, cx,
                                            );
                                            item.focus_handle.focus(window, cx);
                                            cx.stop_propagation();
                                            cx.notify();
                                        },
                                    )),
                            )
                            .child(
                                div()
                                    .debug_selector(move || format!("{}-RESET", field.selector()))
                                    .child(
                                        Button::new(
                                            ("graphite-brightness-contrast-reset", field.index()),
                                            "Reset",
                                        )
                                        .on_click(
                                            cx.listener(move |item, _, window, cx| {
                                                let text = field
                                                    .value(BRIGHTNESS_CONTRAST_TEMPLATE_PARAMETERS)
                                                    .to_string();
                                                item.brightness_contrast_drafts[field.index()] =
                                                    text != item.brightness_contrast_displayed
                                                        [field.index()];
                                                item.brightness_contrast_editors[field.index()]
                                                    .update(cx, |editor, cx| {
                                                        editor.set_text(text, window, cx)
                                                    });
                                                item.commit_brightness_contrast(field, window, cx);
                                            }),
                                        ),
                                    ),
                            )
                    }),
            )
            .when_some(
                self.brightness_contrast_diagnostic(cx).map(str::to_owned),
                |view, diagnostic| {
                    view.child(
                        div()
                            .id("graphite-brightness-contrast-error")
                            .role(gpui::Role::Alert)
                            .debug_selector(|| "GRAPHITE-BRIGHTNESS-CONTRAST-ERROR".into())
                            .child(Label::new(diagnostic).color(Color::Error)),
                    )
                },
            )
    }

    fn cancel_load(&mut self, cx: &mut Context<Self>) {
        if self.pending_scene_preview.take().is_some() {
            self.scene_acceptance_diagnostic =
                Some("Scene preview preparation was cancelled".into());
        }
        self.session.update(cx, |session, cx| session.cancel(cx));
        cx.notify();
    }
}

impl Render for ImportedImageItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        use graphics_core::resources::CpuImageEdit;

        let presentation = self.presentation.clone();
        let max_preview_bytes = self.max_preview_bytes;
        let (
            revoked,
            source_fresh,
            loading,
            operation,
            has_result,
            result_fresh,
            session_diagnostic,
            preparation,
        ) = self.session.update(cx, |session, cx| {
            let result = session.result().cloned();
            let preparation = presentation.update(cx, |presentation, cx| {
                if let Some(preparation) = presentation.refresh_selected_scene(session, window, cx)
                {
                    preparation
                } else if let Some(result) = &result {
                    presentation.refresh_result(
                        session.resource_state(),
                        result,
                        max_preview_bytes,
                        window,
                        cx,
                    )
                } else {
                    presentation.refresh(session.resource_state(), max_preview_bytes, window, cx)
                }
            });
            (
                session.resource_state().is_revoked(),
                session.resource_state().is_fresh(),
                session.resource_state().has_active_request(),
                session.operation(),
                result.is_some(),
                session.result_is_fresh(),
                session.diagnostic().map(|diagnostic| match diagnostic {
                    crate::session::ImageSessionDiagnostic::Resource(error) => error.to_string(),
                    crate::session::ImageSessionDiagnostic::Read(message)
                    | crate::session::ImageSessionDiagnostic::Evaluation(message) => {
                        message.clone()
                    }
                }),
                preparation,
            )
        });
        self.preview_diagnostic = self
            .scene_acceptance_diagnostic
            .clone()
            .or_else(|| preparation.err().map(|error| error.to_string()));
        let (has_image, preview_fresh) = self.presentation.read_with(cx, |presentation, _| {
            (presentation.image().is_some(), presentation.is_fresh())
        });
        let (status, status_selector) = if revoked {
            ("Image unavailable", "GRAPHITE-IMPORTED-REVOKED")
        } else if has_image
            && (!source_fresh
                || !preview_fresh
                || (has_result && !result_fresh)
                || (matches!(operation, Some(CpuImageEdit::BlackAndWhite(_)))
                    && (self.black_and_white_diagnostic().is_some()
                        || self.black_and_white_drafts.iter().any(|draft| *draft)))
                || (matches!(operation, Some(CpuImageEdit::BrightnessContrast(_)))
                    && (self.brightness_contrast_diagnostic(cx).is_some()
                        || self.brightness_contrast_has_draft(cx))))
        {
            ("Last valid image", "GRAPHITE-IMPORTED-LAST-VALID")
        } else if has_image && result_fresh {
            ("Edited image", "GRAPHITE-IMPORTED-RESULT")
        } else if has_image {
            ("Original image", "GRAPHITE-IMPORTED-ORIGINAL")
        } else if loading {
            ("Loading image", "GRAPHITE-IMPORTED-LOADING")
        } else {
            ("No image loaded", "GRAPHITE-IMPORTED-EMPTY")
        };
        v_flex()
            .id("graphite-imported-image-item")
            .role(gpui::Role::Group)
            .aria_label("Imported Graphite image")
            .track_focus(&self.focus_handle)
            .key_context("GraphiteImage")
            .size_full()
            .gap_2()
            .p_4()
            .on_action(
                cx.listener(|this, _: &ShowOriginalImage, _, cx| this.select_operation(None, cx)),
            )
            .on_action(cx.listener(|this, _: &InvertImage, _, cx| {
                this.select_operation(Some(CpuImageEdit::Invert), cx)
            }))
            .on_action(cx.listener(|this, _: &MakeImageOpaque, _, cx| {
                this.select_operation(Some(CpuImageEdit::MakeOpaque), cx)
            }))
            .on_action(cx.listener(|this, _: &CancelImageLoad, _, cx| this.cancel_load(cx)))
            .child(Label::new("Graphite image"))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Button::new("graphite-image-original", "Original")
                            .toggle_state(operation.is_none())
                            .on_click(
                                cx.listener(|this, _, _, cx| this.select_operation(None, cx)),
                            ),
                    )
                    .child(
                        Button::new("graphite-image-invert", "Invert")
                            .toggle_state(operation == Some(CpuImageEdit::Invert))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.select_operation(Some(CpuImageEdit::Invert), cx)
                            })),
                    )
                    .child(
                        Button::new("graphite-image-opaque", "Make opaque")
                            .toggle_state(operation == Some(CpuImageEdit::MakeOpaque))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.select_operation(Some(CpuImageEdit::MakeOpaque), cx)
                            })),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-BLACK-WHITE-SELECT".into())
                            .child(
                                Button::new("graphite-image-black-white", "Black & White")
                                    .toggle_state(matches!(
                                        operation,
                                        Some(CpuImageEdit::BlackAndWhite(_))
                                    ))
                                    .on_click(cx.listener(|item, _, _, cx| {
                                        let parameters =
                                            item.current_black_and_white_parameters(cx);
                                        item.select_operation(
                                            Some(CpuImageEdit::BlackAndWhite(parameters)),
                                            cx,
                                        );
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-BRIGHTNESS-CONTRAST-SELECT".into())
                            .child(
                                Button::new(
                                    "graphite-image-brightness-contrast",
                                    "Brightness/Contrast",
                                )
                                .toggle_state(matches!(
                                    operation,
                                    Some(CpuImageEdit::BrightnessContrast(_))
                                ))
                                .on_click(cx.listener(
                                    |item, _, _, cx| {
                                        let parameters =
                                            item.current_brightness_contrast_parameters(cx);
                                        item.select_operation(
                                            Some(CpuImageEdit::BrightnessContrast(parameters)),
                                            cx,
                                        );
                                    },
                                )),
                            ),
                    )
                    .child(
                        Button::new("graphite-image-cancel", "Cancel")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel_load(cx))),
                    ),
            )
            .when(self.scene_configuration.is_some(), |view| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-IMPORTED-PREPARE-SCENE".into())
                        .child(
                            Button::new("graphite-image-prepare-scene", "Prepare scene")
                                .disabled(self.pending_scene_preview.is_some())
                                .on_click(cx.listener(|item, _, window, cx| {
                                    item.request_scene_preview(window, cx)
                                })),
                        ),
                )
            })
            .when(self.pending_scene_preview.is_some(), |view| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-IMPORTED-SCENE-PENDING".into())
                        .child(Label::new("Preparing scene")),
                )
            })
            .when(
                matches!(operation, Some(CpuImageEdit::BlackAndWhite(_))),
                |view| view.child(self.black_and_white_controls(window, cx)),
            )
            .when(
                matches!(operation, Some(CpuImageEdit::BrightnessContrast(_))),
                |view| view.child(self.brightness_contrast_controls(window, cx)),
            )
            .child(
                div()
                    .debug_selector(move || status_selector.into())
                    .child(Label::new(status)),
            )
            .when_some(session_diagnostic, |view, diagnostic| {
                view.child(
                    div()
                        .id("graphite-imported-session-error")
                        .role(gpui::Role::Alert)
                        .debug_selector(|| "GRAPHITE-IMPORTED-SESSION-ERROR".into())
                        .child(Label::new(diagnostic).color(Color::Error)),
                )
            })
            .when_some(self.preview_diagnostic.clone(), |view, diagnostic| {
                view.child(
                    div()
                        .id("graphite-imported-preview-error")
                        .role(gpui::Role::Alert)
                        .debug_selector(|| "GRAPHITE-IMPORTED-PREVIEW-ERROR".into())
                        .child(Label::new(diagnostic).color(Color::Error)),
                )
            })
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .child(self.presentation.clone()),
            )
    }
}

impl Focusable for ImportedImageItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for ImportedImageItem {}

impl Item for ImportedImageItem {
    type Event = ();

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Graphite image".into()
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Task<Option<Entity<Self>>> {
        gpui::Task::ready(Some(cx.new(|cx| {
            let mut item = Self::from_shared(
                self.session.clone(),
                self.presentation.clone(),
                self.max_preview_bytes,
                window,
                cx,
            );
            item.scene_configuration = self.scene_configuration;
            item
        })))
    }
}

actions!(
    graphite,
    [
        /// Cancels pending evaluation in the focused Graphite Mix item.
        CancelMixEvaluation,
        /// Re-evaluates the focused Graphite Mix item from its current inputs.
        RetryMixEvaluation
    ]
);

pub struct GraphiteMixItem {
    session: Entity<crate::session::GraphiteMixSession>,
    presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
    pub(crate) opacity_editor: Entity<Editor>,
    observed_opacity: u64,
    displayed_opacity: String,
    opacity_has_draft: bool,
    input_diagnostic: Option<String>,
    max_preview_bytes: usize,
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl GraphiteMixItem {
    pub fn from_session(
        session: Entity<crate::session::GraphiteMixSession>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        Self::from_shared(session, presentation, max_preview_bytes, window, cx)
    }

    fn from_shared(
        session: Entity<crate::session::GraphiteMixSession>,
        presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let opacity = session.read(cx).parameters().opacity;
        let opacity_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_text(opacity.to_string(), window, cx);
            editor.set_placeholder_text("Opacity (%)", window, cx);
            editor
        });
        let subscriptions = vec![
            cx.observe(&session, |_, _, cx| cx.notify()),
            cx.observe(&presentation, |_, _, cx| cx.notify()),
            cx.subscribe_in(
                &opacity_editor,
                window,
                |item, editor, event, window, cx| match event {
                    editor::EditorEvent::BufferEdited => {
                        item.opacity_has_draft = editor.read(cx).text(cx) != item.displayed_opacity;
                        cx.notify();
                    }
                    editor::EditorEvent::Blurred => item.commit_opacity(window, cx),
                    _ => {}
                },
            ),
        ];
        Self {
            session,
            presentation,
            opacity_editor,
            observed_opacity: opacity.to_bits(),
            displayed_opacity: opacity.to_string(),
            opacity_has_draft: false,
            input_diagnostic: None,
            max_preview_bytes,
            focus_handle: cx.focus_handle(),
            _subscriptions: subscriptions,
        }
    }

    fn display_committed_opacity(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let opacity = self.session.read(cx).parameters().opacity;
        self.observed_opacity = opacity.to_bits();
        self.displayed_opacity = opacity.to_string();
        self.opacity_has_draft = false;
        self.opacity_editor.update(cx, |editor, cx| {
            editor.set_text(self.displayed_opacity.clone(), window, cx);
        });
    }

    fn commit_opacity(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.opacity_has_draft {
            self.input_diagnostic = None;
            self.display_committed_opacity(window, cx);
            cx.notify();
            return;
        }
        let expression = self.opacity_editor.read(cx).text(cx);
        match graphics_core::numeric_input::evaluate_number_input(
            &expression,
            graphics_core::numeric_input::NumberInputConstraints {
                minimum: Some(0.),
                maximum: Some(100.),
                is_integer: false,
            },
        ) {
            Ok(opacity) => {
                self.input_diagnostic = None;
                self.session.update(cx, |session, cx| {
                    let mut parameters = session.parameters();
                    if parameters.opacity != opacity {
                        parameters.opacity = opacity;
                        session.set_parameters(parameters, cx);
                    }
                });
                self.display_committed_opacity(window, cx);
            }
            Err(error) => self.input_diagnostic = Some(error.to_string()),
        }
        cx.notify();
    }

    pub fn session(&self) -> &Entity<crate::session::GraphiteMixSession> {
        &self.session
    }

    pub fn presentation(&self) -> &Entity<crate::image_preview::CpuPreviewPresentation> {
        &self.presentation
    }
}

impl Render for GraphiteMixItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        use crate::image_preview::{CpuPreviewParent, MixPreviewRequest};

        let (
            parents,
            result,
            parameters,
            current_generation,
            result_generation,
            result_is_fresh,
            pending,
            revoked,
            diagnostic,
        ) = self.session.read_with(cx, |session, cx| {
            (
                session
                    .parents()
                    .each_ref()
                    .map(|parent| CpuPreviewParent::from_state(parent.read(cx).resource_state())),
                session.result(cx).cloned(),
                session.parameters(),
                session.generation(),
                session.result_generation(),
                session.result_is_fresh(cx),
                session.is_pending(),
                session.is_revoked(cx),
                session.diagnostic().map(str::to_owned),
            )
        });
        if self.observed_opacity != parameters.opacity.to_bits()
            && !self.opacity_has_draft
            && !self.opacity_editor.read(cx).is_focused(window)
            && self.input_diagnostic.is_none()
        {
            self.display_committed_opacity(window, cx);
        }
        let preview = self.presentation.update(cx, |presentation, cx| {
            presentation.refresh_mix(
                MixPreviewRequest {
                    parents,
                    result: result.as_ref(),
                    parameters,
                    current_generation,
                    result_generation,
                    result_is_fresh,
                },
                self.max_preview_bytes,
                window,
                cx,
            )
        });
        let session = self.session.downgrade();
        let menu = svg_blend_mode_menu(window, cx, move |mode, _, cx| {
            if let Some(session) = session.upgrade() {
                session.update(cx, |session, cx| {
                    let mut parameters = session.parameters();
                    parameters.blend_mode = mode;
                    session.set_parameters(parameters, cx);
                });
            }
        });
        let presentation = self.presentation.read(cx);
        let (status, selector) = if revoked {
            ("Image unavailable", "GRAPHITE-MIX-REVOKED")
        } else if presentation.image().is_some()
            && (!presentation.is_fresh() || self.input_diagnostic.is_some())
        {
            ("Last valid image", "GRAPHITE-MIX-LAST-VALID")
        } else if presentation.image().is_some() {
            ("Composited image", "GRAPHITE-MIX-RESULT")
        } else if pending {
            ("Evaluating image", "GRAPHITE-MIX-PENDING")
        } else {
            ("Waiting for inputs", "GRAPHITE-MIX-EMPTY")
        };
        let error = self
            .input_diagnostic
            .clone()
            .or(diagnostic)
            .or_else(|| preview.err().map(|error| error.to_string()));
        v_flex()
            .id("graphite-mix-item")
            .role(gpui::Role::Group)
            .aria_label("Graphite image composition")
            .track_focus(&self.focus_handle)
            .key_context("GraphiteMix")
            .size_full()
            .gap_2()
            .p_4()
            .on_action(cx.listener(|item, _: &CancelMixEvaluation, _, cx| {
                item.session.update(cx, |session, cx| session.cancel(cx));
            }))
            .on_action(cx.listener(|item, _: &RetryMixEvaluation, _, cx| {
                item.session.update(cx, |session, cx| session.refresh(cx));
            }))
            .child(Label::new("Graphite Mix"))
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-MIX-BLEND-MODE".into())
                            .child(DropdownMenu::new(
                                "graphite-mix-mode",
                                parameters.blend_mode.to_string(),
                                menu,
                            )),
                    )
                    .child(Label::new("Opacity (%)"))
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-MIX-OPACITY".into())
                            .w_24()
                            .child(self.opacity_editor.clone())
                            .on_action(cx.listener(|item, _: &menu::Confirm, window, cx| {
                                item.commit_opacity(window, cx);
                                window.blur();
                            }))
                            .on_action(cx.listener(|item, _: &menu::Cancel, window, cx| {
                                item.input_diagnostic = None;
                                item.display_committed_opacity(window, cx);
                                window.blur();
                                cx.notify();
                            })),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-MIX-RESET-OPACITY".into())
                            .child(Button::new("graphite-mix-reset", "Reset opacity").on_click(
                                cx.listener(|item, _, window, cx| {
                                    item.opacity_editor.update(cx, |editor, cx| {
                                        editor.set_text("100", window, cx);
                                    });
                                    item.opacity_has_draft = true;
                                    item.commit_opacity(window, cx);
                                }),
                            )),
                    )
                    .child(
                        Button::new("graphite-mix-cancel", "Cancel").on_click(cx.listener(
                            |item, _, _, cx| {
                                item.session.update(cx, |session, cx| session.cancel(cx));
                            },
                        )),
                    )
                    .child(
                        Button::new("graphite-mix-retry", "Retry").on_click(cx.listener(
                            |item, _, _, cx| {
                                item.session.update(cx, |session, cx| session.refresh(cx));
                            },
                        )),
                    ),
            )
            .child(
                div()
                    .debug_selector(move || selector.into())
                    .child(Label::new(status)),
            )
            .when_some(error, |view, error| {
                view.child(
                    div()
                        .id("graphite-mix-error")
                        .role(gpui::Role::Alert)
                        .debug_selector(|| "GRAPHITE-MIX-ERROR".into())
                        .child(Label::new(error).color(Color::Error)),
                )
            })
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .child(self.presentation.clone()),
            )
    }
}

impl Focusable for GraphiteMixItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for GraphiteMixItem {}

impl Item for GraphiteMixItem {
    type Event = ();

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Graphite Mix".into()
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Task<Option<Entity<Self>>> {
        gpui::Task::ready(Some(cx.new(|cx| {
            Self::from_shared(
                self.session.clone(),
                self.presentation.clone(),
                self.max_preview_bytes,
                window,
                cx,
            )
        })))
    }
}

fn svg_blend_mode_menu(
    window: &mut Window,
    cx: &mut App,
    on_select: impl Fn(graphics_core::resources::CpuBlendMode, &mut Window, &mut App) + 'static,
) -> Entity<ContextMenu> {
    let on_select = std::rc::Rc::new(on_select);
    ContextMenu::build(window, cx, move |mut menu, _, _| {
        for mode in graphics_core::resources::CpuBlendMode::list_svg_subset()
            .into_iter()
            .flatten()
        {
            let mode = *mode;
            let on_select = on_select.clone();
            menu = menu.entry(mode.to_string(), None, move |window, cx| {
                on_select(mode, window, cx);
            });
        }
        menu
    })
}

#[derive(Clone, Copy)]
enum ArtboardField {
    LocationX,
    LocationY,
    Width,
    Height,
    Background,
    ImageXX,
    ImageXY,
    ImageYX,
    ImageYY,
    ImageX,
    ImageY,
}

impl ArtboardField {
    const ALL: [Self; 11] = [
        Self::LocationX,
        Self::LocationY,
        Self::Width,
        Self::Height,
        Self::Background,
        Self::ImageXX,
        Self::ImageXY,
        Self::ImageYX,
        Self::ImageYY,
        Self::ImageX,
        Self::ImageY,
    ];

    fn index(self) -> usize {
        match self {
            Self::LocationX => 0,
            Self::LocationY => 1,
            Self::Width => 2,
            Self::Height => 3,
            Self::Background => 4,
            Self::ImageXX => 5,
            Self::ImageXY => 6,
            Self::ImageYX => 7,
            Self::ImageYY => 8,
            Self::ImageX => 9,
            Self::ImageY => 10,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::LocationX => "X (px)",
            Self::LocationY => "Y (px)",
            Self::Width => "Width (px)",
            Self::Height => "Height (px)",
            Self::Background => "Background (gamma-sRGB RGBA)",
            Self::ImageXX => "Image X axis X",
            Self::ImageXY => "Image X axis Y",
            Self::ImageYX => "Image Y axis X",
            Self::ImageYY => "Image Y axis Y",
            Self::ImageX => "Image X (px)",
            Self::ImageY => "Image Y (px)",
        }
    }

    fn selector(self) -> &'static str {
        match self {
            Self::LocationX => "GRAPHITE-ARTBOARD-X",
            Self::LocationY => "GRAPHITE-ARTBOARD-Y",
            Self::Width => "GRAPHITE-ARTBOARD-WIDTH",
            Self::Height => "GRAPHITE-ARTBOARD-HEIGHT",
            Self::Background => "GRAPHITE-ARTBOARD-BACKGROUND",
            Self::ImageXX => "GRAPHITE-ARTBOARD-IMAGE-XX",
            Self::ImageXY => "GRAPHITE-ARTBOARD-IMAGE-XY",
            Self::ImageYX => "GRAPHITE-ARTBOARD-IMAGE-YX",
            Self::ImageYY => "GRAPHITE-ARTBOARD-IMAGE-YY",
            Self::ImageX => "GRAPHITE-ARTBOARD-IMAGE-X",
            Self::ImageY => "GRAPHITE-ARTBOARD-IMAGE-Y",
        }
    }

    fn placement_index(self) -> Option<usize> {
        match self {
            Self::ImageXX => Some(0),
            Self::ImageXY => Some(1),
            Self::ImageYX => Some(2),
            Self::ImageYY => Some(3),
            Self::ImageX => Some(4),
            Self::ImageY => Some(5),
            _ => None,
        }
    }

    fn text(
        self,
        parameters: graphics_core::evaluate::CpuSvgArtboardParameters,
        placement: Option<graphite_graph_craft::document::value::DAffine2>,
    ) -> String {
        match self {
            Self::LocationX => parameters.location.x.to_string(),
            Self::LocationY => parameters.location.y.to_string(),
            Self::Width => parameters.dimensions.x.to_string(),
            Self::Height => parameters.dimensions.y.to_string(),
            Self::Background => parameters
                .background
                .to_gamma_srgb_channels()
                .map(|channel| channel.to_string())
                .join(", "),
            _ => self
                .placement_index()
                .and_then(|index| {
                    placement.and_then(|placement| placement.to_cols_array().get(index).copied())
                })
                .map(|value| value.to_string())
                .unwrap_or_default(),
        }
    }
}

struct PendingItemArtboardPreview {
    generation: u64,
    _task: gpui::Task<()>,
}

pub struct GraphiteArtboardItem {
    session: Entity<crate::session::GraphiteArtboardSession>,
    presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
    max_preview_bytes: usize,
    automatic_preview: bool,
    linked_document: Option<Entity<crate::interop::GraphiteLinkedDocument>>,
    delivery_cancellation: Option<std::sync::Arc<sim_tensor::CancellationToken>>,
    delivery_status: Option<String>,
    delivery_diagnostic: Option<String>,
    png_limits: Option<(usize, usize)>,
    prepared_png: Option<crate::session::PreparedArtboardPngAsset>,
    png_diagnostic: Option<String>,
    png_generation: u64,
    pending_png: Option<PendingItemArtboardPreview>,
    pub(crate) artboard_editors: [Entity<Editor>; 11],
    displayed: [String; 11],
    drafts: [bool; 11],
    field_diagnostics: [Option<String>; 11],
    preview_diagnostic: Option<String>,
    request_generation: u64,
    pending_preview: Option<PendingItemArtboardPreview>,
    focus_handle: FocusHandle,
    _subscriptions: Vec<Subscription>,
}

impl GraphiteArtboardItem {
    pub fn from_session(
        session: Entity<crate::session::GraphiteArtboardSession>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        Self::from_shared(session, presentation, max_preview_bytes, window, cx)
    }

    fn from_shared(
        session: Entity<crate::session::GraphiteArtboardSession>,
        presentation: Entity<crate::image_preview::CpuPreviewPresentation>,
        max_preview_bytes: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let parameters = session.read(cx).authoring_parameters();
        let placement = session.read(cx).authoring_image_placement();
        let displayed = ArtboardField::ALL.map(|field| field.text(parameters, placement));
        let artboard_editors = ArtboardField::ALL.map(|field| {
            cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_text(field.text(parameters, placement), window, cx);
                editor.set_placeholder_text(field.label(), window, cx);
                editor
            })
        });
        let mut subscriptions = vec![
            cx.observe(&presentation, |_, _, cx| cx.notify()),
            cx.subscribe_in(
                &session,
                window,
                |item, _, _: &crate::session::ArtboardSessionEvent, window, cx| {
                    item.synchronize_fields(window, cx);
                    item.refresh_resource_authority(window, cx);
                    cx.notify();
                },
            ),
        ];
        for (field, editor) in ArtboardField::ALL.into_iter().zip(&artboard_editors) {
            subscriptions.push(cx.subscribe_in(
                editor,
                window,
                move |item, editor, event, window, cx| match event {
                    editor::EditorEvent::BufferEdited => {
                        item.drafts[field.index()] =
                            editor.read(cx).text(cx) != item.displayed[field.index()];
                        cx.notify();
                    }
                    editor::EditorEvent::Blurred => item.commit_field(field, window, cx),
                    _ => {}
                },
            ));
        }
        Self {
            session,
            presentation,
            max_preview_bytes,
            automatic_preview: false,
            linked_document: None,
            delivery_cancellation: None,
            delivery_status: None,
            delivery_diagnostic: None,
            png_limits: None,
            prepared_png: None,
            png_diagnostic: None,
            png_generation: 0,
            pending_png: None,
            artboard_editors,
            displayed,
            drafts: [false; 11],
            field_diagnostics: std::array::from_fn(|_| None),
            preview_diagnostic: None,
            request_generation: 0,
            pending_preview: None,
            focus_handle: cx.focus_handle(),
            _subscriptions: subscriptions,
        }
    }

    pub(crate) fn set_linked_document(
        &mut self,
        linked: crate::interop::GraphiteLinkedDocument,
        cx: &mut Context<Self>,
    ) {
        let linked = cx.new(|_| linked);
        self.set_shared_linked_document(linked, cx);
    }

    fn set_shared_linked_document(
        &mut self,
        linked: Entity<crate::interop::GraphiteLinkedDocument>,
        cx: &mut Context<Self>,
    ) {
        self._subscriptions
            .push(cx.observe(&linked, |_, _, cx| cx.notify()));
        self.linked_document = Some(linked);
        cx.notify();
    }

    pub fn saved_sim_source<'a>(
        &'a self,
        cx: &'a App,
    ) -> Option<&'a sim_runtime::ProjectEditableSourceReference> {
        self.linked_document
            .as_ref()
            .and_then(|linked| linked.read(cx).saved_source.as_ref())
    }

    pub fn input_version<'a>(
        &'a self,
        cx: &'a App,
    ) -> Option<&'a sim_runtime::ProjectAssetVersionReference> {
        self.linked_document
            .as_ref()
            .map(|linked| &linked.read(cx).input_version.reference)
    }

    pub fn delivery_status(&self) -> Option<&str> {
        self.delivery_status.as_deref()
    }
    pub fn delivery_diagnostic(&self) -> Option<&str> {
        self.delivery_diagnostic.as_deref()
    }

    fn delivery_ready(&self, cx: &App) -> anyhow::Result<()> {
        let linked = self
            .linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Open a project PNG to enable .sim saving and export."))?
            .read(cx);
        anyhow::ensure!(
            linked.project.current(cx) && !self.session.read(cx).resource_is_revoked(),
            "The project asset permission changed. Reopen the document."
        );
        anyhow::ensure!(
            self.session.read(cx).gesture().is_none()
                && !self.drafts.iter().any(|draft| *draft)
                && !self.field_diagnostics.iter().any(Option::is_some),
            "Apply or discard the current composition edits before saving or exporting."
        );
        Ok(())
    }

    pub fn save_sim(
        &mut self,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<sim_runtime::OutputCommitReceipt> {
        self.delivery_ready(cx)?;
        let linked = self
            .linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .read(cx)
            .clone();
        let mut container = linked.container.clone();
        let session = self.session.read(cx);
        let revision = session.revision();
        let payload = graphics_core::formats::capture_artboard_engine_payload(
            session.document(),
            linked.resource_id,
            sim_runtime::MAX_SIM_DOCUMENT_BYTES,
        )?;
        let engine = container
            .documents
            .get_mut(linked.engine_index)
            .ok_or_else(|| anyhow::anyhow!("The .sim engine document is missing."))?;
        let document_id = engine.document_id;
        *engine = sim_runtime::SimEngineDocument::new(
            engine.document_id,
            engine.engine.clone(),
            engine.source_revision.clone(),
            engine.serialization_format.clone(),
            String::from_utf8(payload.serialized_bytes().to_vec())?,
            vec![linked.input_version.reference.clone()],
        );
        let metadata = serde_json::json!({
            "schema": "sim.editable-document.v1",
            "document_id": document_id,
            "input_version": linked.input_version.reference,
            "source_revision": graphics_core::resources::GRAPHITE_IMAGE_SOURCE_REVISION,
            "revision": revision
        });
        let proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            format!("graphite/{document_id}/editable"),
            "sim",
            0,
            0,
            0,
            container.encode()?,
        )?
        .with_projection_metadata(serde_json::to_vec(&metadata)?)?;
        let cancellation = sim_tensor::CancellationToken::default();
        // The final authority check and canonical commit cannot yield to a profile
        // replacement or edit on GPUI's foreground thread.
        let receipt = linked.project.publish(proposal, &cancellation)?;
        let source = sim_runtime::ProjectEditableSourceReference {
            identity: receipt.operation().identity.clone(),
            content_sha256: receipt.operation().sha256.clone(),
        };
        self.linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .update(cx, |linked, cx| {
                linked.container = container;
                linked.saved_source = Some(source);
                linked.saved_revision = Some(revision);
                cx.notify();
            });
        self.delivery_status = Some(format!(
            "Saved .sim: {}",
            receipt.operation().identity.to_reference()?
        ));
        self.delivery_diagnostic = None;
        cx.notify();
        Ok(receipt)
    }

    pub fn export_baked_png(
        &mut self,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<gpui::Task<anyhow::Result<sim_runtime::OutputCommitReceipt>>> {
        self.delivery_ready(cx)?;
        anyhow::ensure!(
            self.delivery_cancellation.is_none(),
            "An export is already running."
        );
        let preview = self
            .png_preview(cx)
            .ok_or_else(|| anyhow::anyhow!("Wait for the current preview before exporting."))?;
        let revision = self.session.read(cx).revision();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        let project = self
            .linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .read(cx)
            .project
            .clone();
        project.track_operation(&cancellation, cx)?;
        let preparation = self.session.update(cx, |session, cx| {
            session.prepare_png(preview, 128 * 1024 * 1024, 64 * 1024 * 1024, cx)
        })?;
        self.delivery_cancellation = Some(cancellation.clone());
        self.delivery_status = Some("Exporting to project Output/graphite".into());
        self.delivery_diagnostic = None;
        cx.notify();
        Ok(cx.spawn(async move |item, cx| {
            let outcome = async {
                let prepared = preparation.await?;
                item.update(cx, |item, cx| {
                    item.complete_png_export(prepared, revision, &cancellation, cx)
                })?
            }
            .await;
            if let Err(error) = item.update(cx, |item, cx| {
                if item
                    .delivery_cancellation
                    .as_ref()
                    .is_some_and(|current| std::sync::Arc::ptr_eq(current, &cancellation))
                {
                    item.delivery_cancellation = None;
                    if let Err(error) = &outcome {
                        item.delivery_diagnostic = Some(error.to_string());
                        item.delivery_status = None;
                    }
                    cx.notify();
                }
            }) {
                log::debug!("The PNG delivery Item was released: {error}");
            }
            outcome
        }))
    }

    fn complete_png_export(
        &mut self,
        prepared: crate::session::PreparedArtboardPngAsset,
        revision: u64,
        cancellation: &std::sync::Arc<sim_tensor::CancellationToken>,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<sim_runtime::OutputCommitReceipt> {
        self.delivery_ready(cx)?;
        anyhow::ensure!(
            !cancellation.is_cancelled()
                && self
                    .delivery_cancellation
                    .as_ref()
                    .is_some_and(|current| std::sync::Arc::ptr_eq(current, cancellation)),
            "Export was cancelled."
        );
        anyhow::ensure!(
            self.session.read(cx).revision() == revision
                && self.session.read(cx).png_is_current(&prepared),
            "The composition changed. Export the refreshed result."
        );
        if self
            .linked_document
            .as_ref()
            .is_none_or(|linked| linked.read(cx).saved_revision != Some(revision))
        {
            self.save_sim(cx)?;
        }
        let linked = self
            .linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .read(cx)
            .clone();
        let source = linked
            .saved_source
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Save the editable source before exporting."))?;
        let document_id = linked
            .container
            .documents
            .get(linked.engine_index)
            .ok_or_else(|| anyhow::anyhow!("The engine document is missing."))?
            .document_id;
        let proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            format!("graphite/{document_id}/baked"),
            "png",
            0,
            prepared.png().width(),
            prepared.png().height(),
            prepared.png().encoded_bytes().to_vec(),
        )?;
        let prepared = crate::storage::PreparedArtboardPngProposal::new(
            self.session.read(cx),
            prepared,
            proposal,
        )?;
        let mut provenance: serde_json::Value =
            serde_json::from_slice(prepared.proposal().projection_metadata())?;
        provenance["input_versions"] =
            serde_json::to_value(vec![linked.input_version.reference.clone()])?;
        provenance["editable_source"] = serde_json::to_value(&source)?;
        provenance["delivery_profile"] =
            serde_json::json!({"format":"png","color_space":"sRGB","alpha":"straight"});
        let proposal = sim_runtime::OutputProposal::new(
            uuid::Uuid::new_v4(),
            sim_runtime::assets::AssetNamespace::Output,
            format!("graphite/{document_id}/baked"),
            "png",
            0,
            prepared.proposal().width(),
            prepared.proposal().height(),
            prepared.proposal().content().to_vec(),
        )?
        .with_projection_metadata(serde_json::to_vec(&provenance)?)?;
        // Keep the canonical asset-to-writer lock order. No foreground
        // yield may occur between the currentness check and publication.
        let mut assets = linked.project.assets.try_lock().map_err(|_| {
            anyhow::anyhow!("The project asset service is busy. Retry after the current job.")
        })?;
        let mut committer = linked.project.committer.try_lock().map_err(|_| {
            anyhow::anyhow!("The project publisher is busy. Retry after the current job.")
        })?;
        assets.resolve_project_asset_version(
            &linked.input_version.reference,
            &linked.project.output_reader,
            cancellation,
            64 * 1024 * 1024,
        )?;
        let receipts = committer.commit_proposal_batch_and_register(
            &[proposal],
            chrono::Local::now().fixed_offset(),
            &mut assets,
            &linked.project.publisher,
            cancellation,
        )?;
        let receipt = receipts
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("The publisher returned no committed PNG."))?;
        let version = assets
            .register_project_png_version(
                &receipt,
                receipt.operation().identity.clone(),
                document_id,
                vec![linked.input_version.reference.clone()],
                Some(source),
                provenance,
                &linked.project.output_reader,
                &linked.project.publisher,
                cancellation,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "PNG bytes committed at {}, but project-version registration failed: {error}. Export is incomplete.",
                    receipt.operation().identity.relative_path.display()
                )
            })?;
        drop(committer);
        drop(assets);
        self.linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .update(cx, |linked, cx| {
                if !linked.container.build_outputs.contains(&version.reference) {
                    linked.container.build_outputs.push(version.reference);
                    cx.notify();
                }
            });
        self.save_sim(cx).map_err(|error| {
            anyhow::anyhow!(
                "PNG committed at {}, but saving its .sim build-output link failed: {error}. Delivery is incomplete.",
                receipt.operation().identity.relative_path.display()
            )
        })?;
        self.delivery_status = Some(format!(
            "Exported baked PNG: {}",
            receipt.operation().identity.to_reference()?
        ));
        Ok(receipt)
    }

    pub fn update_input_version(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> anyhow::Result<gpui::Task<anyhow::Result<bool>>> {
        self.delivery_ready(cx)?;
        anyhow::ensure!(
            self.delivery_cancellation.is_none(),
            "A delivery operation is already running."
        );
        let linked = self
            .linked_document
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
            .read(cx)
            .clone();
        let revision = self.session.read(cx).revision();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        linked.project.track_operation(&cancellation, cx)?;
        let background = cx.background_spawn({
            let cancellation = cancellation.clone();
            async move { crate::interop::prepare_linked_input_update(&linked, &cancellation) }
        });
        self.delivery_cancellation = Some(cancellation.clone());
        self.delivery_status = Some("Checking the original PNG for a new version".into());
        self.delivery_diagnostic = None;
        cx.notify();
        Ok(cx.spawn_in(window, async move |item, cx| {
            let result = async {
                let (version, source) = background.await?;
                item.update_in(cx, |item, window, cx| {
                    item.delivery_ready(cx)?;
                    anyhow::ensure!(
                        !cancellation.is_cancelled()
                            && item.delivery_cancellation.as_ref().is_some_and(|current| {
                                std::sync::Arc::ptr_eq(current, &cancellation)
                            }),
                        "Updating the input was cancelled."
                    );
                    anyhow::ensure!(
                        item.session.read(cx).revision() == revision,
                        "The composition changed while checking the input. Retry; edits are preserved."
                    );
                    let unchanged = item.linked_document.as_ref().is_some_and(|linked| {
                        linked.read(cx).input_version.reference == version.reference
                    });
                    let changed = if unchanged {
                        false
                    } else {
                        item.session.update(cx, |session, cx| {
                            session.replace_encoded_source(revision, source, cx)
                        })?
                    };
                    item.linked_document.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("The linked document was closed."))?
                        .update(cx, |linked, cx| {
                            if let Some(provenance) = linked.container.provenance.as_object_mut() {
                                provenance.insert("input_version".into(), serde_json::json!(version.reference));
                                provenance.insert("producer_provenance".into(), version.provenance.clone());
                                provenance.insert("producer_editable_source".into(), serde_json::json!(version.editable_source));
                            }
                            linked.input_version = version;
                            if changed {
                                linked.saved_revision = None;
                            }
                            cx.notify();
                        });
                    if changed {
                        item.prepared_png = None;
                        item.session.update(cx, |session, cx| session.cancel_png(cx));
                        item.refresh_automatically(window, cx);
                    }
                    item.delivery_status = Some(if changed {
                        "Updated input version; composition edits preserved"
                    } else {
                        "The pinned input version is already current"
                    }.into());
                    Ok::<_, anyhow::Error>(changed)
                })?
            }.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                if item.delivery_cancellation.as_ref().is_some_and(|current| {
                    std::sync::Arc::ptr_eq(current, &cancellation)
                }) {
                    item.delivery_cancellation = None;
                    if let Err(error) = &result {
                        item.delivery_status = None;
                        item.delivery_diagnostic = Some(error.to_string());
                    }
                    cx.notify();
                }
            }) {
                log::debug!("The input update Item was released: {error}");
            }
            result
        }))
    }

    pub fn cancel_delivery(&mut self, cx: &mut Context<Self>) {
        if let Some(cancellation) = self.delivery_cancellation.take() {
            cancellation.cancel();
            self.delivery_status = None;
            self.delivery_diagnostic = Some("Export was cancelled.".into());
            cx.notify();
        }
    }

    pub fn session(&self) -> &Entity<crate::session::GraphiteArtboardSession> {
        &self.session
    }

    pub fn presentation(&self) -> &Entity<crate::image_preview::CpuPreviewPresentation> {
        &self.presentation
    }

    pub(crate) fn enable_automatic_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.automatic_preview = true;
        self.refresh_automatically(window, cx);
    }

    fn refresh_automatically(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let session = self.session.read(cx);
        if !self.automatic_preview
            || session.resource_is_revoked()
            || session.gesture().is_some()
            || self.drafts.iter().any(|draft| *draft)
            || self.field_diagnostics.iter().any(Option::is_some)
            || session
                .preview()
                .is_some_and(|preview| session.preview_is_current(preview))
        {
            return;
        }
        self.request_preview(window, cx);
    }

    pub fn configure_png_limits(
        &mut self,
        max_pixel_bytes: usize,
        max_encoded_bytes: usize,
        cx: &mut Context<Self>,
    ) {
        let limits = Some((max_pixel_bytes, max_encoded_bytes));
        if self.png_limits != limits {
            self.pending_png = None;
            self.prepared_png = None;
            self.png_diagnostic = None;
            self.png_limits = limits;
            cx.notify();
        }
    }

    pub(crate) fn prepared_png(&self) -> Option<&crate::session::PreparedArtboardPngAsset> {
        self.prepared_png.as_ref()
    }

    pub(crate) fn png_is_pending(&self) -> bool {
        self.pending_png.is_some()
    }

    fn png_preview(&self, cx: &App) -> Option<crate::session::PreparedArtboardPreview> {
        if self.session.read(cx).gesture().is_some()
            || self.drafts.iter().any(|draft| *draft)
            || self.field_diagnostics.iter().any(Option::is_some)
        {
            return None;
        }
        let session = self.session.read(cx);
        let preview = session.preview()?;
        let image = self.presentation.read(cx).image()?;
        (session.preview_is_current(preview) && std::sync::Arc::ptr_eq(image, preview.image()))
            .then(|| preview.clone())
    }

    fn request_png(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.pending_png.is_some() {
            return;
        }
        let Some((max_pixel_bytes, max_encoded_bytes)) = self.png_limits else {
            self.png_diagnostic = Some("PNG preparation limits have not been supplied".into());
            cx.notify();
            return;
        };
        let Some(preview) = self.png_preview(cx) else {
            self.png_diagnostic = Some("Prepare a current Artboard preview before its PNG".into());
            cx.notify();
            return;
        };
        let Some(generation) = self.png_generation.checked_add(1) else {
            self.png_diagnostic = Some("Artboard PNG request generation is exhausted".into());
            cx.notify();
            return;
        };
        let preparation = match self.session.update(cx, |session, cx| {
            session.prepare_png(preview, max_pixel_bytes, max_encoded_bytes, cx)
        }) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.png_diagnostic = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self.png_generation = generation;
        self.png_diagnostic = None;
        let task = cx.spawn_in(window, async move |item, cx| {
            let outcome = preparation.await;
            if let Err(error) = item.update_in(cx, |item, _, cx| {
                if !item
                    .pending_png
                    .as_ref()
                    .is_some_and(|pending| pending.generation == generation)
                {
                    return;
                }
                item.pending_png = None;
                match outcome {
                    Ok(prepared) if item.session.read(cx).png_is_current(&prepared) => {
                        item.prepared_png = Some(prepared);
                        item.png_diagnostic = None;
                    }
                    Ok(_) => item.png_diagnostic = Some("The prepared PNG was superseded".into()),
                    Err(error) => item.png_diagnostic = Some(error.to_string()),
                }
                cx.notify();
            }) {
                log::debug!("Artboard PNG Item or window was released: {error}");
            }
        });
        self.pending_png = Some(PendingItemArtboardPreview {
            generation,
            _task: task,
        });
        cx.notify();
    }

    fn cancel_png(&mut self, cx: &mut Context<Self>) {
        let pending = self.pending_png.take();
        let owns_completed = self
            .prepared_png
            .as_ref()
            .is_some_and(|prepared| self.session.read(cx).png_is_current(prepared));
        if pending.is_none() && owns_completed {
            self.session
                .update(cx, |session, cx| session.cancel_png(cx));
        }
        if pending.is_some() || owns_completed {
            self.png_diagnostic = Some("PNG preparation was cancelled".into());
        }
        drop(pending);
        cx.notify();
    }

    fn field_description(&self, field: ArtboardField, cx: &App) -> String {
        let session = self.session.read(cx);
        let committed = field.text(session.parameters(), session.image_placement());
        let stored = if session.gesture().is_some() {
            format!(
                "Unapplied batch value: {}. Committed value: {committed}.",
                self.displayed[field.index()]
            )
        } else {
            format!("Stored value: {committed}.")
        };
        let error = self.field_diagnostics[field.index()]
            .as_ref()
            .map_or_else(String::new, |error| format!("{}: {error}. ", field.label()));
        format!(
            "{error}{stored} Enter applies this field; Escape restores the last accepted field value. Tab and Shift-Tab move between controls."
        )
    }

    fn display_field(&mut self, field: ArtboardField, window: &mut Window, cx: &mut Context<Self>) {
        let session = self.session.read(cx);
        let text = field.text(
            session.authoring_parameters(),
            session.authoring_image_placement(),
        );
        self.displayed[field.index()] = text.clone();
        self.drafts[field.index()] = false;
        self.artboard_editors[field.index()]
            .update(cx, |editor, cx| editor.set_text(text, window, cx));
    }

    fn synchronize_fields(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let parameters = self.session.read(cx).authoring_parameters();
        let placement = self.session.read(cx).authoring_image_placement();
        for field in ArtboardField::ALL {
            let editor = &self.artboard_editors[field.index()];
            if !self.drafts[field.index()]
                && self.field_diagnostics[field.index()].is_none()
                && !editor.read(cx).is_focused(window)
                && editor.read(cx).text(cx) != field.text(parameters, placement)
            {
                self.display_field(field, window, cx);
            }
        }
    }

    fn commit_field(&mut self, field: ArtboardField, window: &mut Window, cx: &mut Context<Self>) {
        if !self.drafts[field.index()] {
            return;
        }
        let text = self.artboard_editors[field.index()].read(cx).text(cx);
        if let Some(index) = field.placement_index() {
            let outcome = graphics_core::numeric_input::evaluate_number_input(
                &text,
                graphics_core::numeric_input::NumberInputConstraints::default(),
            )
            .map_err(|error| error.to_string())
            .and_then(|value| {
                let mut columns = self
                    .session
                    .read(cx)
                    .authoring_image_placement()
                    .ok_or_else(|| {
                        crate::session::ArtboardSessionError::MissingEncodedPngInput.to_string()
                    })?
                    .to_cols_array();
                let component = columns
                    .get_mut(index)
                    .ok_or_else(|| "Unknown image placement component".to_string())?;
                *component = value;
                let placement =
                    graphite_graph_craft::document::value::DAffine2::from_cols_array(&columns);
                self.session
                    .update(cx, |session, cx| {
                        if let Some(gesture) = session.gesture() {
                            session.stage_gesture(
                                &gesture,
                                session.authoring_parameters(),
                                Some(placement),
                                cx,
                            )
                        } else {
                            session.apply_image_placement(session.revision(), placement, cx)
                        }
                    })
                    .map_err(|error| error.to_string())
            });
            self.finish_field_commit(field, outcome, window, cx);
            return;
        }

        let mut parameters = self.session.read(cx).authoring_parameters();
        let parsed = match field {
            ArtboardField::Background => {
                crate::session::parse_color(&text, "Background").map(|[red, green, blue, alpha]| {
                    parameters.background = graphite_core_types::Color::from_gamma_srgb_channels(
                        red, green, blue, alpha,
                    );
                })
            }
            _ => graphics_core::numeric_input::evaluate_number_input(
                &text,
                graphics_core::numeric_input::NumberInputConstraints {
                    minimum: Some(-9_007_199_254_740_992.),
                    maximum: Some(9_007_199_254_740_992.),
                    is_integer: true,
                },
            )
            .map(|value| match field {
                ArtboardField::LocationX => parameters.location.x = value,
                ArtboardField::LocationY => parameters.location.y = value,
                ArtboardField::Width => parameters.dimensions.x = value,
                ArtboardField::Height => parameters.dimensions.y = value,
                _ => {}
            })
            .map_err(|error| error.to_string()),
        };
        let outcome = parsed.and_then(|()| {
            self.session
                .update(cx, |session, cx| {
                    if let Some(gesture) = session.gesture() {
                        session
                            .stage_gesture(
                                &gesture,
                                parameters,
                                session.authoring_image_placement(),
                                cx,
                            )
                            .map_err(|error| error.to_string())
                    } else {
                        session
                            .apply(session.revision(), parameters, session.configuration(), cx)
                            .map_err(|error| error.to_string())
                    }
                })
                .map_err(|error| error.to_string())
        });
        self.finish_field_commit(field, outcome, window, cx);
    }

    fn finish_field_commit(
        &mut self,
        field: ArtboardField,
        outcome: Result<bool, String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match outcome {
            Ok(changed) => {
                self.field_diagnostics[field.index()] = None;
                if changed && self.session.read(cx).gesture().is_none() {
                    self.pending_preview = None;
                    self.pending_png = None;
                    self.preview_diagnostic = None;
                }
                self.display_field(field, window, cx);
                self.refresh_presentation(cx);
                self.refresh_automatically(window, cx);
            }
            Err(error) => self.field_diagnostics[field.index()] = Some(error),
        }
        cx.notify();
    }

    fn begin_gesture(&mut self, cx: &mut Context<Self>) {
        let outcome = self.session.update(cx, |session, cx| {
            session.begin_gesture(session.revision(), cx)
        });
        self.preview_diagnostic = outcome.err().map(|error| error.to_string());
        cx.notify();
    }

    fn finish_gesture(&mut self, commit: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(gesture) = self.session.read(cx).gesture() else {
            return;
        };
        let outcome = self.session.update(cx, |session, cx| {
            if commit {
                session.commit_gesture(&gesture, cx)
            } else {
                session.cancel_gesture(&gesture, cx).map(|()| false)
            }
        });
        match outcome {
            Ok(changed) => {
                self.preview_diagnostic = None;
                if changed {
                    self.pending_preview = None;
                    self.pending_png = None;
                }
                if !commit {
                    for field in ArtboardField::ALL {
                        self.field_diagnostics[field.index()] = None;
                        self.display_field(field, window, cx);
                    }
                }
                self.synchronize_fields(window, cx);
                self.refresh_presentation(cx);
                self.refresh_automatically(window, cx);
            }
            Err(error) => self.preview_diagnostic = Some(error.to_string()),
        }
        cx.notify();
    }

    fn toggle_clip(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let outcome = self.session.update(cx, |session, cx| {
            let mut parameters = session.authoring_parameters();
            parameters.clip = !parameters.clip;
            if let Some(gesture) = session.gesture() {
                session
                    .stage_gesture(
                        &gesture,
                        parameters,
                        session.authoring_image_placement(),
                        cx,
                    )
                    .map_err(|error| error.to_string())
            } else {
                session
                    .apply(session.revision(), parameters, session.configuration(), cx)
                    .map_err(|error| error.to_string())
            }
        });
        match outcome {
            Ok(changed) => {
                if changed && self.session.read(cx).gesture().is_none() {
                    self.pending_preview = None;
                    self.pending_png = None;
                    self.preview_diagnostic = None;
                }
                self.refresh_presentation(cx);
                self.refresh_automatically(window, cx);
            }
            Err(error) => self.preview_diagnostic = Some(error.to_string()),
        }
        cx.notify();
    }

    fn refresh_presentation(&self, cx: &mut Context<Self>) {
        if self.presentation.read(cx).image().is_none() {
            return;
        }
        let presentation = self.presentation.clone();
        self.session.update(cx, |session, cx| {
            presentation.update(cx, |presentation, cx| {
                presentation.refresh_artboard_status(session, cx);
            });
        });
    }

    fn refresh_resource_authority(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.session.read(cx).resource_is_revoked() {
            self.pending_preview = None;
            self.pending_png = None;
            self.prepared_png = None;
            self.preview_diagnostic =
                Some(crate::session::ArtboardSessionError::ResourceRevoked.to_string());
            self.png_diagnostic = self.preview_diagnostic.clone();
        }
        let presentation = self.presentation.clone();
        self.session.update(cx, |session, cx| {
            presentation.update(cx, |presentation, cx| {
                presentation.refresh_artboard_resource_status(session, window, cx);
            });
        });
    }

    fn request_preview(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.pending_preview.is_some() || self.session.read(cx).is_pending() {
            return;
        }
        let Some(generation) = self.request_generation.checked_add(1) else {
            self.preview_diagnostic = Some("Artboard Item request generation is exhausted".into());
            cx.notify();
            return;
        };
        let max_preview_bytes = self.max_preview_bytes;
        let preparation = match self.session.update(cx, |session, cx| {
            session.request_preview(max_preview_bytes, cx)
        }) {
            Ok(preparation) => preparation,
            Err(error) => {
                self.preview_diagnostic = Some(error.to_string());
                cx.notify();
                return;
            }
        };
        self.request_generation = generation;
        self.preview_diagnostic = None;
        let task = cx.spawn_in(window, async move |item, cx| {
            let outcome = preparation.await;
            if let Err(error) = item.update_in(cx, |item, window, cx| {
                if !item
                    .pending_preview
                    .as_ref()
                    .is_some_and(|pending| pending.generation == generation)
                {
                    return;
                }
                item.pending_preview = None;
                match outcome {
                    Ok(preview) => {
                        let presentation = item.presentation.clone();
                        let acceptance = item.session.update(cx, |session, cx| {
                            presentation.update(cx, |presentation, cx| {
                                presentation.accept_artboard_preview(session, &preview, window, cx)
                            })
                        });
                        item.preview_diagnostic = acceptance.err().map(|error| error.to_string());
                    }
                    Err(error) => item.preview_diagnostic = Some(error.to_string()),
                }
                cx.notify();
            }) {
                log::debug!("Artboard preview Item or window was released: {error}");
            }
        });
        self.pending_preview = Some(PendingItemArtboardPreview {
            generation,
            _task: task,
        });
        cx.notify();
    }

    fn cancel_preview(&mut self, cx: &mut Context<Self>) {
        self.session.update(cx, |session, cx| session.cancel(cx));
        self.pending_preview = None;
        self.pending_png = None;
        self.preview_diagnostic = None;
        self.refresh_presentation(cx);
        cx.notify();
    }
}

impl Render for GraphiteArtboardItem {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.synchronize_fields(window, cx);
        self.refresh_resource_authority(window, cx);
        let parameters = self.session.read(cx).authoring_parameters();
        let has_image = self.session.read(cx).authoring_image_placement().is_some();
        let gesture_active = self.session.read(cx).gesture().is_some();
        let can_apply = !self.drafts.iter().any(|draft| *draft)
            && !self.field_diagnostics.iter().any(Option::is_some);
        let pending = self.pending_preview.is_some() || self.session.read(cx).is_pending();
        let diagnostic = self
            .field_diagnostics
            .iter()
            .find_map(Clone::clone)
            .or_else(|| self.preview_diagnostic.clone())
            .or_else(|| self.session.read(cx).diagnostic().map(ToString::to_string));
        let png_enabled =
            self.png_limits.is_some() && !self.png_is_pending() && self.png_preview(cx).is_some();
        let png_cancel_enabled = self.pending_png.is_some()
            || self
                .prepared_png
                .as_ref()
                .is_some_and(|prepared| self.session.read(cx).png_is_current(prepared));
        let png_status = self.prepared_png().map(|prepared| {
            let current =
                self.session.read(cx).png_is_current(prepared) && self.png_preview(cx).is_some();
            let prefix = if current {
                "PNG prepared"
            } else {
                "Last prepared PNG"
            };
            (
                format!(
                    "{prefix}: {} × {}",
                    prepared.png().width(),
                    prepared.png().height()
                ),
                current,
            )
        });
        let presentation = self.presentation.read(cx);
        let (status, selector) = if presentation.image().is_some()
            && (!presentation.is_fresh()
                || gesture_active
                || self.drafts.iter().any(|draft| *draft)
                || self.field_diagnostics.iter().any(Option::is_some))
        {
            ("Last valid image", "GRAPHITE-ARTBOARD-LAST-VALID")
        } else if presentation.image().is_some() {
            ("Artboard preview", "GRAPHITE-ARTBOARD-RESULT")
        } else if gesture_active {
            ("Unapplied Artboard edits", "GRAPHITE-ARTBOARD-UNAPPLIED")
        } else if pending {
            ("Preparing Artboard", "GRAPHITE-ARTBOARD-PENDING")
        } else {
            ("Prepare a preview", "GRAPHITE-ARTBOARD-EMPTY")
        };
        v_flex()
            .id("graphite-artboard-item")
            .role(gpui::Role::Group)
            .aria_label("Graphite Artboard")
            .track_focus(&self.focus_handle.clone().tab_index(-1).tab_stop(false))
            .tab_group()
            .tab_index(0isize)
            .tab_stop(false)
            .key_context("GraphiteArtboard")
            .on_action(cx.listener(|_, _: &FocusNextArtboardControl, window, cx| {
                window.focus_next(cx);
                cx.stop_propagation();
            }))
            .on_action(
                cx.listener(|_, _: &FocusPreviousArtboardControl, window, cx| {
                    window.focus_prev(cx);
                    cx.stop_propagation();
                }),
            )
            .on_action(cx.listener(|_, _: &editor::actions::Tab, window, cx| {
                window.focus_next(cx);
                cx.stop_propagation();
            }))
            .on_action(cx.listener(|_, _: &editor::actions::Backtab, window, cx| {
                window.focus_prev(cx);
                cx.stop_propagation();
            }))
            .size_full()
            .gap_2()
            .p_4()
            .child(Label::new("Graphite Artboard"))
            .child(Label::new(
                "Background channels: red, green, blue, alpha (gamma-sRGB).",
            ))
            .children(
                ArtboardField::ALL
                    .into_iter()
                    .filter(|field| has_image || field.placement_index().is_none())
                    .map(|field| {
                        let editor = self.artboard_editors[field.index()].clone();
                        let focus = editor
                            .focus_handle(cx)
                            .tab_index(field.index() as isize)
                            .tab_stop(true);
                        let value = editor.read(cx).text(cx);
                        let description = self.field_description(field, cx);
                        let error = self.field_diagnostics[field.index()].clone();
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex().gap_2().child(Label::new(field.label())).child(
                                    div()
                                        .id(field.selector())
                                        .debug_selector(move || field.selector().into())
                                        .role(gpui::Role::TextInput)
                                        .aria_label(field.label())
                                        .aria_value(value)
                                        .aria_description(description)
                                        .track_focus(&focus)
                                        .tab_index(field.index() as isize)
                                        .border_1()
                                        .border_color(cx.theme().colors().border)
                                        .when(focus.is_focused(window), |input| {
                                            input.border_color(cx.theme().colors().border_focused)
                                        })
                                        .w_64()
                                        .child(editor)
                                        .on_action(cx.listener(
                                            move |item, _: &menu::Confirm, window, cx| {
                                                item.commit_field(field, window, cx);
                                                item.focus_handle.focus(window, cx);
                                                cx.stop_propagation();
                                            },
                                        ))
                                        .on_action(cx.listener(
                                            move |item, _: &menu::Cancel, window, cx| {
                                                item.field_diagnostics[field.index()] = None;
                                                item.display_field(field, window, cx);
                                                item.focus_handle.focus(window, cx);
                                                cx.stop_propagation();
                                                cx.notify();
                                            },
                                        )),
                                ),
                            )
                            .when(self.drafts[field.index()], |row| {
                                row.child(Label::new(format!(
                                    "Last accepted field value: {}",
                                    self.displayed[field.index()]
                                )))
                            })
                            .when_some(error, |row, error| {
                                row.child(
                                    ui::diagnostic_alert(
                                        format!("{}-error", field.selector()),
                                        format!("{} error: {error}", field.label()),
                                        Label::new(format!("{}: {error}", field.label()))
                                            .color(Color::Error),
                                    )
                                    .debug_selector(move || format!("{}-ERROR", field.selector())),
                                )
                            })
                    }),
            )
            .child(
                h_flex()
                    .gap_2()
                    .when(!gesture_active, |row| {
                        row.child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-BEGIN-EDITS".into())
                                .child(
                                    Button::new("graphite-artboard-begin-edits", "Batch edit")
                                        .tab_index(11isize)
                                        .on_click(
                                            cx.listener(|item, _, _, cx| item.begin_gesture(cx)),
                                        ),
                                ),
                        )
                    })
                    .when(gesture_active, |row| {
                        row.child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-APPLY-EDITS".into())
                                .child(
                                    Button::new("graphite-artboard-apply-edits", "Apply edits")
                                        .tab_index(12isize)
                                        .disabled(!can_apply)
                                        .on_click(cx.listener(|item, _, window, cx| {
                                            item.finish_gesture(true, window, cx)
                                        })),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-DISCARD-EDITS".into())
                                .child(
                                    Button::new("graphite-artboard-discard-edits", "Discard edits")
                                        .tab_index(13isize)
                                        .on_click(cx.listener(|item, _, window, cx| {
                                            item.finish_gesture(false, window, cx)
                                        })),
                                ),
                        )
                    }),
            )
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-ARTBOARD-CLIP".into())
                            .child(
                                Button::new(
                                    "graphite-artboard-clip",
                                    if parameters.clip {
                                        "Clip: On"
                                    } else {
                                        "Clip: Off"
                                    },
                                )
                                .tab_index(14isize)
                                .toggle_state(parameters.clip)
                                .on_click(
                                    cx.listener(|item, _, window, cx| item.toggle_clip(window, cx)),
                                ),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-ARTBOARD-PREPARE".into())
                            .child(
                                Button::new("graphite-artboard-prepare", "Prepare preview")
                                    .tab_index(15isize)
                                    .disabled(pending || gesture_active)
                                    .on_click(cx.listener(|item, _, window, cx| {
                                        item.request_preview(window, cx)
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .debug_selector(|| "GRAPHITE-ARTBOARD-CANCEL".into())
                            .child(
                                Button::new("graphite-artboard-cancel", "Cancel")
                                    .tab_index(16isize)
                                    .on_click(
                                        cx.listener(|item, _, _, cx| item.cancel_preview(cx)),
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .debug_selector(move || selector.into())
                    .child(Label::new(status)),
            )
            .when_some(diagnostic, |view, diagnostic| {
                view.child(
                    ui::diagnostic_alert(
                        "graphite-artboard-error",
                        diagnostic.clone(),
                        Label::new(diagnostic).color(Color::Error),
                    )
                    .debug_selector(|| "GRAPHITE-ARTBOARD-ERROR".into()),
                )
            })
            .when(self.png_limits.is_some(), |view| {
                view.child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-PREPARE-PNG".into())
                                .child(
                                    Button::new("graphite-artboard-prepare-png", "Prepare PNG")
                                        .tab_index(17isize)
                                        .disabled(!png_enabled)
                                        .on_click(cx.listener(|item, _, window, cx| {
                                            item.request_png(window, cx)
                                        })),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-CANCEL-PNG".into())
                                .child(
                                    Button::new("graphite-artboard-cancel-png", "Cancel PNG")
                                        .tab_index(18isize)
                                        .disabled(!png_cancel_enabled)
                                        .on_click(
                                            cx.listener(|item, _, _, cx| item.cancel_png(cx)),
                                        ),
                                ),
                        ),
                )
            })
            .when(self.png_is_pending(), |view| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-ARTBOARD-PNG-PENDING".into())
                        .child(Label::new("Preparing PNG")),
                )
            })
            .when_some(png_status, |view, (status, current)| {
                view.child(
                    div()
                        .debug_selector(move || {
                            if current {
                                "GRAPHITE-ARTBOARD-PNG-RESULT".into()
                            } else {
                                "GRAPHITE-ARTBOARD-PNG-LAST-VALID".into()
                            }
                        })
                        .child(Label::new(status)),
                )
            })
            .when_some(self.png_diagnostic.clone(), |view, diagnostic| {
                view.child(
                    div()
                        .id("graphite-artboard-png-error")
                        .role(gpui::Role::Alert)
                        .debug_selector(|| "GRAPHITE-ARTBOARD-PNG-ERROR".into())
                        .child(Label::new(diagnostic).color(Color::Error)),
                )
            })
            .when(self.linked_document.is_some(), |view| {
                view.child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-SAVE-SIM".into())
                                .child(
                                    Button::new("graphite-artboard-save-sim", "Save .sim")
                                        .tab_index(19isize)
                                        .on_click(cx.listener(|item, _, _, cx| {
                                            if let Err(error) = item.save_sim(cx) {
                                                item.delivery_diagnostic = Some(error.to_string());
                                                cx.notify();
                                            }
                                        })),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-EXPORT-PNG".into())
                                .child(
                                    Button::new("graphite-artboard-export-png", "Export baked PNG")
                                        .tab_index(20isize)
                                        .disabled(self.delivery_cancellation.is_some())
                                        .on_click(cx.listener(|item, _, _, cx| {
                                            match item.export_baked_png(cx) {
                                                Ok(task) => {
                                                    task.detach();
                                                }
                                                Err(error) => {
                                                    item.delivery_diagnostic =
                                                        Some(error.to_string());
                                                    cx.notify();
                                                }
                                            }
                                        })),
                                ),
                        )
                        .child(
                            div()
                                .debug_selector(|| "GRAPHITE-ARTBOARD-CANCEL-DELIVERY".into())
                                .child(
                                    Button::new(
                                        "graphite-artboard-cancel-delivery",
                                        "Cancel export",
                                    )
                                    .tab_index(21isize)
                                    .disabled(self.delivery_cancellation.is_none())
                                    .on_click(
                                        cx.listener(|item, _, _, cx| item.cancel_delivery(cx)),
                                    ),
                                ),
                        ),
                )
            })
            .when(self.linked_document.is_some(), |view| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-ARTBOARD-UPDATE-INPUT".into())
                        .child(
                            Button::new("graphite-artboard-update-input", "Update input version")
                                .tab_index(22isize)
                                .disabled(self.delivery_cancellation.is_some())
                                .on_click(cx.listener(|item, _, window, cx| {
                                    match item.update_input_version(window, cx) {
                                        Ok(task) => task.detach(),
                                        Err(error) => {
                                            item.delivery_diagnostic = Some(error.to_string());
                                            cx.notify();
                                        }
                                    }
                                })),
                        ),
                )
            })
            .when_some(self.delivery_status.clone(), |view, status| {
                view.child(
                    div()
                        .debug_selector(|| "GRAPHITE-ARTBOARD-DELIVERY-RESULT".into())
                        .child(Label::new(status)),
                )
            })
            .when_some(self.delivery_diagnostic.clone(), |view, error| {
                view.child(
                    div()
                        .id("graphite-artboard-delivery-error")
                        .role(gpui::Role::Alert)
                        .debug_selector(|| "GRAPHITE-ARTBOARD-DELIVERY-ERROR".into())
                        .child(Label::new(error).color(Color::Error)),
                )
            })
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.))
                    .child(self.presentation.clone()),
            )
    }
}

impl Focusable for GraphiteArtboardItem {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<()> for GraphiteArtboardItem {}

impl Item for GraphiteArtboardItem {
    type Event = ();

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Graphite Artboard".into()
    }

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::Task<Option<Entity<Self>>> {
        gpui::Task::ready(Some(cx.new(|cx| {
            let mut item = Self::from_shared(
                self.session.clone(),
                self.presentation.clone(),
                self.max_preview_bytes,
                window,
                cx,
            );
            item.png_limits = self.png_limits;
            item.automatic_preview = self.automatic_preview;
            if let Some(linked) = &self.linked_document {
                item.set_shared_linked_document(linked.clone(), cx);
            }
            item
        })))
    }
}

impl Drop for GraphiteArtboardItem {
    fn drop(&mut self) {
        if let Some(cancellation) = self.delivery_cancellation.take() {
            cancellation.cancel();
        }
    }
}

#[cfg(test)]
mod artboard_keyboard_tests {
    use super::*;
    use crate::conformance::artboard_workspace_controls::{
        click, native_artboard, paint, parameters,
    };
    use gpui::{TestAppContext, VisualTestContext};
    use std::sync::Arc;

    fn accessibility_tree(window: &mut VisualTestContext) -> serde_json::Value {
        let json = window
            .update(|window, _| window.debug_a11y_tree_json())
            .expect("actual activated native accessibility tree");
        serde_json::from_str(&json).expect("native debug tree JSON")
    }

    fn activate_accessibility(window: &mut VisualTestContext) {
        let handle = window.window_handle();
        assert!(
            window
                .cx
                .simulate_accessibility_activation(handle)
                .is_some(),
            "the actual platform activation callback must be registered"
        );
        window.run_until_parked();
        paint(window);
    }

    fn field_node(tree: &serde_json::Value, label: &str) -> serde_json::Value {
        tree["nodes"]
            .as_object()
            .expect("actual native nodes")
            .values()
            .find(|node| node["aria"]["role"] == "TextInput" && node["aria"]["label"] == label)
            .expect("actual named field node")
            .clone()
    }

    fn focused_label(window: &mut VisualTestContext) -> String {
        let tree = accessibility_tree(window);
        let focused = tree["gpui_focus"]
            .as_str()
            .expect("actual focused accessibility node");
        tree["nodes"][focused]["aria"]["label"]
            .as_str()
            .expect("actual focused control label")
            .to_owned()
    }

    fn key(window: &mut VisualTestContext, keystroke: &str) {
        window.simulate_keystrokes(keystroke);
        let keystroke = gpui::Keystroke::parse(keystroke).expect("actual native key release");
        window.update(|window, cx| {
            window.dispatch_event(
                gpui::PlatformInput::KeyUp(gpui::KeyUpEvent { keystroke }),
                cx,
            );
        });
        window.run_until_parked();
        paint(window);
    }

    fn focus_control(window: &mut VisualTestContext, label: &str) {
        for _ in 0..24 {
            key(window, "tab");
            if focused_label(window) == label {
                return;
            }
        }
        panic!("native Tab did not reach {label}");
    }

    fn focus_item(item: &Entity<GraphiteArtboardItem>, window: &mut VisualTestContext) {
        let focus = item.read_with(window, |item, cx| item.focus_handle(cx));
        window.update(|window, cx| focus.focus(window, cx));
        paint(window);
    }

    #[gpui::test]
    async fn at_r24_artboard_native_tab_shift_tab_exposes_each_actual_editor_and_value(
        cx: &mut TestAppContext,
    ) {
        let (item, _workspace, window) = native_artboard(cx, parameters(), 31 * 17 * 4).await;
        activate_accessibility(window);
        let session = item.read_with(window, |item, _| item.session().clone());
        let network = session.read_with(window, |session, _| session.document().network().clone());
        focus_item(&item, window);
        for field in ArtboardField::ALL.into_iter().take(5) {
            key(window, "tab");
            assert_eq!(focused_label(window), field.label());
            let editor = item.read_with(window, |item, _| {
                item.artboard_editors[field.index()].clone()
            });
            window.update(|window, cx| assert!(editor.read(cx).is_focused(window)));
            let tree = accessibility_tree(window);
            let node = field_node(&tree, field.label());
            assert_eq!(
                node["aria"]["value"],
                editor.read_with(window, |editor, cx| editor.text(cx))
            );
            assert!(
                node["aria"]["description"]
                    .as_str()
                    .expect("field description")
                    .contains("Escape restores the last accepted field value")
            );
        }
        for field in ArtboardField::ALL.into_iter().take(4).rev() {
            key(window, "shift-tab");
            assert_eq!(focused_label(window), field.label());
        }
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), 0);
            assert_eq!(session.parameters(), parameters());
            assert_eq!(session.document().network(), &network);
            assert!(!session.is_pending());
            assert!(session.preview().is_none());
        });
        assert!(
            accessibility_tree(window)["nodes"]
                .as_object()
                .expect("actual nodes")
                .values()
                .all(|node| node["aria"]["label"] != ArtboardField::ImageXX.label())
        );
    }

    #[gpui::test]
    async fn at_r24_artboard_native_invalid_enter_escape_error_value_and_current_frame_recovery(
        cx: &mut TestAppContext,
    ) {
        let (item, _workspace, window) = native_artboard(cx, parameters(), 31 * 17 * 4).await;
        click(window, "GRAPHITE-ARTBOARD-PREPARE");
        let session = item.read_with(window, |item, _| item.session().clone());
        let original = session.read_with(window, |session, _| {
            session.preview().expect("original real frame").clone()
        });
        activate_accessibility(window);
        focus_item(&item, window);
        key(window, "tab");
        assert_eq!(focused_label(window), ArtboardField::LocationX.label());
        let editor = item.read_with(window, |item, _| item.artboard_editors[0].clone());
        editor.update_in(window, |editor, window, cx| editor.set_text("", window, cx));
        window.simulate_input("invalid");
        window.run_until_parked();
        paint(window);
        key(window, "enter");
        let error = item.read_with(window, |item, _| {
            item.field_diagnostics[0]
                .clone()
                .expect("exact field error")
        });
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-X-ERROR").is_some());
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        let tree = accessibility_tree(window);
        let node = field_node(&tree, ArtboardField::LocationX.label());
        assert_eq!(node["aria"]["value"], "invalid");
        let description = node["aria"]["description"]
            .as_str()
            .expect("actual field error description");
        assert!(description.contains(&error));
        assert!(description.contains("Stored value: 2."));
        assert!(
            tree["nodes"]
                .as_object()
                .expect("actual nodes")
                .values()
                .any(|node| node["aria"]["role"] == "Alert"
                    && node["aria"]["label"] == format!("X (px) error: {error}"))
        );
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), 0);
            assert_eq!(session.parameters(), parameters());
            assert!(session.preview_is_current(&original));
        });
        assert!(Arc::ptr_eq(
            original.image(),
            &item.read_with(window, |item, cx| item
                .presentation()
                .read(cx)
                .image()
                .expect("last valid frame")
                .clone())
        ));
        key(window, "tab");
        assert_eq!(focused_label(window), ArtboardField::LocationX.label());
        key(window, "escape");
        assert_eq!(editor.read_with(window, |editor, cx| editor.text(cx)), "2");
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-X-ERROR").is_none());
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
        let tree = accessibility_tree(window);
        let node = field_node(&tree, ArtboardField::LocationX.label());
        assert_eq!(node["aria"]["value"], "2");
        assert!(
            !node["aria"]["description"]
                .as_str()
                .expect("restored description")
                .contains(&error)
        );
        assert!(
            !tree["nodes"]
                .as_object()
                .expect("actual nodes")
                .values()
                .any(|node| node["aria"]["label"]
                    .as_str()
                    .is_some_and(|label| label.starts_with("X (px) error:")))
        );
        key(window, "tab");
        editor.update_in(window, |editor, window, cx| editor.set_text("", window, cx));
        window.simulate_input("2+3");
        key(window, "enter");
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), 1);
            assert_eq!(session.parameters().location.x.to_bits(), 5_f64.to_bits());
            assert!(!session.preview_is_current(&original));
        });
        assert!(
            window
                .debug_bounds("GRAPHITE-ARTBOARD-LAST-VALID")
                .is_some()
        );
        focus_control(window, "Prepare preview");
        key(window, "enter");
        session.read_with(window, |session, _| {
            let current = session
                .preview()
                .expect("actual keyboard-requested successor");
            assert!(session.preview_is_current(current));
            assert!(!Arc::ptr_eq(current.image(), original.image()));
            assert_ne!(current.image().as_bytes(0), original.image().as_bytes(0));
            assert_eq!(
                current.output().parameters().location.x.to_bits(),
                5_f64.to_bits()
            );
        });
        assert!(window.debug_bounds("GRAPHITE-ARTBOARD-RESULT").is_some());
    }

    #[gpui::test]
    async fn at_r24_linked_png_native_keyboard_placement_save_and_actual_export(
        cx: &mut TestAppContext,
    ) {
        use crate::png_workflow::tests::{
            native_workspace, open_requested_png, output_records, register_fixture_workflow,
        };
        use crate::resources::tests::{host_reader_fixture, indexed_request_png};
        let fixture = host_reader_fixture().expect("real authorized host reader");
        let png = indexed_request_png(&fixture, 3).expect("real indexed PNG");
        let (workspace, window) = native_workspace(cx).await;
        let (reader, _committer, _publisher) = register_fixture_workflow(&fixture, &png, window);
        let item = open_requested_png(&workspace, window);
        let session = item.read_with(window, |item, _| item.session().clone());
        let original = session.read_with(window, |session, _| {
            let preview = session
                .preview()
                .expect("initial real automatic PNG preview");
            assert!(session.preview_is_current(preview));
            assert_eq!(
                preview.image().size(0),
                gpui::size(gpui::DevicePixels(1920), gpui::DevicePixels(1080))
            );
            preview.clone()
        });
        let source = original
            .output()
            .encoded_png_input()
            .expect("actual original source")
            .source()
            .clone();
        let version = item.read_with(window, |item, cx| {
            item.input_version(cx)
                .expect("actual immutable input link")
                .clone()
        });
        activate_accessibility(window);
        focus_item(&item, window);
        for field in ArtboardField::ALL {
            key(window, "tab");
            assert_eq!(focused_label(window), field.label());
            let editor = item.read_with(window, |item, _| {
                item.artboard_editors[field.index()].clone()
            });
            window.update(|window, cx| assert!(editor.read(cx).is_focused(window)));
            assert_eq!(
                field_node(&accessibility_tree(window), field.label())["aria"]["value"],
                editor.read_with(window, |editor, cx| editor.text(cx))
            );
        }
        key(window, "shift-tab");
        assert_eq!(focused_label(window), ArtboardField::ImageX.label());
        let editor = item.read_with(window, |item, _| item.artboard_editors[9].clone());
        editor.update_in(window, |editor, window, cx| editor.set_text("", window, cx));
        window.simulate_input("12");
        key(window, "enter");
        session.read_with(window, |session, _| {
            assert_eq!(session.revision(), 1);
            assert_eq!(
                session
                    .image_placement()
                    .expect("original affine")
                    .translation
                    .x
                    .to_bits(),
                12_f64.to_bits()
            );
            let current = session
                .preview()
                .expect("automatic refresh after native Enter");
            assert!(session.preview_is_current(current));
            assert!(!session.preview_is_current(&original));
            assert_ne!(current.image().as_bytes(0), original.image().as_bytes(0));
            assert!(Arc::ptr_eq(
                current
                    .output()
                    .encoded_png_input()
                    .expect("same immutable input")
                    .source(),
                &source
            ));
        });
        focus_control(window, "Save .sim");
        key(window, "enter");
        item.read_with(window, |item, cx| {
            assert!(item.delivery_diagnostic().is_none());
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Saved .sim:"))
            );
            assert!(item.saved_sim_source(cx).is_some());
            assert_eq!(item.input_version(cx), Some(&version));
        });
        focus_control(window, "Export baked PNG");
        key(window, "enter");
        item.read_with(window, |item, _| {
            assert!(item.delivery_diagnostic().is_none());
            assert!(
                item.delivery_status()
                    .is_some_and(|status| status.starts_with("Exported baked PNG:"))
            );
        });
        let outputs = output_records(&fixture, &reader, "/baked");
        assert_eq!(outputs.len(), 1);
        let bytes = fixture
            .assets
            .lock()
            .expect("actual shared catalog")
            .read_verified(
                &outputs.first().expect("actual committed output").identity,
                &reader,
                &sim_tensor::CancellationToken::default(),
                128 * 1024 * 1024,
            )
            .expect("actual committed PNG destination bytes");
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .expect("actual baked PNG")
            .to_rgba8();
        assert_eq!(decoded.dimensions(), (1920_u32, 1080_u32));
        let mut frame = session.read_with(window, |session, _| {
            session
                .preview()
                .expect("current real frame")
                .image()
                .as_bytes(0)
                .expect("actual straight BGRA bytes")
                .to_vec()
        });
        let (pixels, remainder) = frame.as_chunks_mut::<4>();
        assert!(remainder.is_empty());
        for pixel in pixels {
            pixel.swap(0, 2);
        }
        assert_eq!(decoded.as_raw(), &frame);
        assert_eq!(
            fixture
                .assets
                .lock()
                .expect("actual service")
                .read_verified(
                    &png.asset_identity,
                    &fixture.input_authorization,
                    &sim_tensor::CancellationToken::default(),
                    16 * 1024
                )
                .expect("original Input remains exact"),
            png.encoded
        );
        assert!(session.read_with(window, |session, _| {
            session.preview_is_current(session.preview().expect("current receipt"))
        }));
    }
}
