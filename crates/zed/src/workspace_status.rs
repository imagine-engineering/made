use gpui::{AppContext as _, Context, Window};
use image_viewer::ImageInfo;
use language_tools::lsp_button::{self, LspButton};
use ui::PopoverMenuHandle;
use workspace::{AppState, Workspace};

pub(crate) fn initialize(
    app_state: &AppState,
    workspace: &mut Workspace,
    window: &mut Window,
    cx: &mut Context<Workspace>,
) {
    let edit_prediction_menu_handle = PopoverMenuHandle::default();
    let edit_prediction_ui = cx.new(|cx| {
        edit_prediction_ui::EditPredictionButton::new(
            app_state.fs.clone(),
            app_state.user_store.clone(),
            edit_prediction_menu_handle.clone(),
            workspace.project().clone(),
            cx,
        )
    });
    workspace.register_action({
        move |_, _: &edit_prediction_ui::ToggleMenu, window, cx| {
            edit_prediction_menu_handle.toggle(window, cx);
        }
    });

    let search_button = cx.new(|_| search::search_status_button::SearchButton::new());
    let diagnostic_summary =
        cx.new(|cx| diagnostics::items::DiagnosticIndicator::new(workspace, cx));
    let active_file_name = cx.new(|_| workspace::active_file_name::ActiveFileName::new());
    let activity_indicator = activity_indicator::ActivityIndicator::new(
        workspace,
        workspace.project().read(cx).languages().clone(),
        window,
        cx,
    );
    let active_buffer_encoding =
        cx.new(|_| encoding_selector::ActiveBufferEncoding::new(workspace));
    let active_buffer_language =
        cx.new(|_| language_selector::ActiveBufferLanguage::new(workspace));
    let active_toolchain_language =
        cx.new(|cx| toolchain_selector::ActiveToolchain::new(workspace, window, cx));
    let vim_mode_indicator = cx.new(|cx| vim::ModeIndicator::new(window, cx));
    let image_info = cx.new(|_cx| ImageInfo::new(workspace));

    let lsp_button_menu_handle = PopoverMenuHandle::default();
    let lsp_button =
        cx.new(|cx| LspButton::new(workspace, lsp_button_menu_handle.clone(), window, cx));
    workspace.register_action({
        move |_, _: &lsp_button::ToggleMenu, window, cx| {
            lsp_button_menu_handle.toggle(window, cx);
        }
    });

    let cursor_position = cx.new(|_| go_to_line::cursor_position::CursorPosition::new(workspace));
    let line_ending_indicator = cx.new(|_| line_ending_selector::LineEndingIndicator::default());
    let git_blame_status = cx.new(|_| git_ui::GitBlameStatus::default());
    #[cfg(feature = "agentic-tools")]
    let merge_conflict_indicator = cx.new(|cx| git_ui::MergeConflictIndicator::new(workspace, cx));
    workspace.status_bar().update(cx, |status_bar, cx| {
        status_bar.add_left_item(search_button, window, cx);
        status_bar.add_left_item(lsp_button, window, cx);
        status_bar.add_left_item(diagnostic_summary, window, cx);
        status_bar.add_left_item(active_file_name, window, cx);
        status_bar.add_left_item(git_blame_status, window, cx);
        #[cfg(feature = "agentic-tools")]
        status_bar.add_left_item(merge_conflict_indicator, window, cx);
        status_bar.add_left_item(activity_indicator, window, cx);
        status_bar.add_right_item(edit_prediction_ui, window, cx);
        status_bar.add_right_item(active_buffer_encoding, window, cx);
        status_bar.add_right_item(active_buffer_language, window, cx);
        status_bar.add_right_item(active_toolchain_language, window, cx);
        status_bar.add_right_item(line_ending_indicator, window, cx);
        status_bar.add_right_item(vim_mode_indicator, window, cx);
        status_bar.add_right_item(cursor_position, window, cx);
        status_bar.add_right_item(image_info, window, cx);
    });
}
