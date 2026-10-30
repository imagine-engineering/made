use gpui::{Context, Focusable, Window, actions};

use crate::{Workspace, WorkspacePresentation};

actions!(
    workspace,
    [
        /// Switches the active workspace to the editor presentation.
        SwitchToEditorWorkspace
    ]
);

#[cfg(feature = "multiplayer-tools")]
actions!(
    workspace,
    [
        /// Switches the active workspace to the multiplayer presentation.
        SwitchToCollaborativeWorkspace
    ]
);

impl Workspace {
    pub(super) fn switch_to_editor_workspace(
        &mut self,
        _: &SwitchToEditorWorkspace,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.switch_workspace_presentation(WorkspacePresentation::Editor, window, cx);
    }

    #[cfg(feature = "multiplayer-tools")]
    pub(super) fn switch_to_collaborative_workspace(
        &mut self,
        _: &SwitchToCollaborativeWorkspace,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.switch_workspace_presentation(WorkspacePresentation::Collaborative, window, cx);
    }

    fn switch_workspace_presentation(
        &mut self,
        presentation: WorkspacePresentation,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let fs = self.project().read(cx).fs().clone();
        settings::update_settings_file(fs, cx, move |settings, _cx| {
            settings.workspace.workspace_presentation = Some(presentation);
        });

        if self.workspace_presentation == presentation {
            return;
        }

        self.workspace_presentation = presentation;
        #[cfg(feature = "multiplayer-tools")]
        self.synchronize_collaborative_participant_surfaces(cx);
        match presentation {
            WorkspacePresentation::Editor => self.active_pane().focus_handle(cx).focus(window, cx),
            #[cfg(feature = "multiplayer-tools")]
            WorkspacePresentation::Collaborative => {
                self.release_empty_collaborative_native_editor(window, cx);
                self.collaborative_workspace
                    .read(cx)
                    .focus_handle(cx)
                    .focus(window, cx)
            }
            #[cfg(not(feature = "multiplayer-tools"))]
            WorkspacePresentation::Collaborative => {
                self.active_pane().focus_handle(cx).focus(window, cx)
            }
        }
        self.serialize_workspace(window, cx);
        #[cfg(feature = "multiplayer-tools")]
        cx.emit(crate::Event::PresentationChanged);
        cx.notify();
    }
}
