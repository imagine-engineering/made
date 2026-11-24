mod call_stats_modal;
pub mod channel_view;
pub mod notifications;

use anyhow::Context as _;
use call::ActiveCall;
use std::{rc::Rc, sync::Arc};

use gpui::TaskExt as _;
use gpui::{
    App, ClipboardItem, Pixels, PlatformDisplay, Size, WindowBackgroundAppearance, WindowBounds,
    WindowDecorations, WindowKind, WindowOptions, point,
};
use release_channel::ReleaseChannel;
use ui::px;
use util::ResultExt;
use workspace::{
    AppState, CopyRoomId, Deafen, LeaveCall, Mute, OpenChannelNotes, OpenChannelNotesById,
    ScreenShare, ShareProject, Workspace, notifications::DetachAndPromptErr,
};

use channel_view::ChannelView;

pub fn init(app_state: &Arc<AppState>, cx: &mut App) {
    call_stats_modal::init(cx);
    channel_view::init(cx);
    register_call_actions(cx);
    notifications::init(app_state, cx);
    title_bar::init(cx);
}

fn register_call_actions(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|_, _: &OpenChannelNotes, window, cx| {
            let channel_id = ActiveCall::global(cx)
                .read(cx)
                .room()
                .and_then(|room| room.read(cx).channel_id());

            if let Some(channel_id) = channel_id {
                let workspace = cx.entity();
                window.defer(cx, move |window, cx| {
                    ChannelView::open(channel_id, None, workspace, window, cx)
                        .detach_and_log_err(cx)
                });
            }
        });
        workspace.register_action(|_, action: &OpenChannelNotesById, window, cx| {
            let channel_id = client::ChannelId(action.channel_id);
            let workspace = cx.entity();
            window.defer(cx, move |window, cx| {
                ChannelView::open(channel_id, None, workspace, window, cx).detach_and_log_err(cx);
            });
        });
        // TODO: make it possible to bind this one to a held key for push to talk?
        // how to make "toggle_on_modifiers_press" contextual?
        workspace.register_action(|_, _: &Mute, _, cx| title_bar::collab::toggle_mute(cx));
        workspace.register_action(|_, _: &Deafen, _, cx| title_bar::collab::toggle_deafen(cx));
        workspace.register_action(|_, _: &LeaveCall, window, cx| {
            ActiveCall::global(cx)
                .update(cx, |call, cx| call.hang_up(cx))
                .detach_and_prompt_err("Failed to hang up", window, cx, |_, _, _| None);
        });
        workspace.register_action(|workspace, _: &CopyRoomId, window, cx| {
            use workspace::notifications::{NotificationId, NotifyTaskExt as _};

            struct RoomIdCopiedToast;

            if let Some(room) = ActiveCall::global(cx).read(cx).room() {
                let room_id = room.read(cx).room_id();
                let workspace_handle = cx.weak_entity();
                cx.spawn(async move |workspace, cx| {
                    let room_id = room_id.await.context("Failed to get livekit room")?;
                    workspace.update(cx, |workspace, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(room_id));
                        workspace.show_toast(
                            workspace::Toast::new(
                                NotificationId::unique::<RoomIdCopiedToast>(),
                                "Room ID copied to clipboard",
                            )
                            .autohide(),
                            cx,
                        );
                    })
                })
                .detach_and_notify_err(workspace_handle, window, cx);
            } else {
                workspace.show_error("There’s no active call; join one first.", cx);
            }
        });
        workspace.register_action(|workspace, _: &ShareProject, window, cx| {
            let project = workspace.project().clone();
            let workspace = cx.weak_entity();
            window.defer(cx, move |_window, cx| {
                ActiveCall::global(cx).update(cx, move |call, cx| {
                    if let Some(room) = call.room() {
                        if room.read(cx).is_sharing_project() {
                            call.unshare_project(project, cx).ok();
                        } else {
                            call.share_project(project, cx).detach_and_log_err(cx);
                        }
                    } else {
                        workspace
                            .update(cx, |workspace, cx| {
                                workspace.show_error(
                            "Join or start a collaboration call before sharing this project.",
                            cx,
                        );
                            })
                            .log_err();
                    }
                });
            });
        });
        // TODO(jk): Is this action ever triggered?
        workspace.register_action(|_, _: &ScreenShare, window, cx| {
            let room = ActiveCall::global(cx).read(cx).room().cloned();
            if let Some(room) = room {
                window.defer(cx, move |_window, cx| {
                    room.update(cx, |room, cx| {
                        if room.is_sharing_screen() {
                            room.unshare_screen(true, cx).ok();
                        } else {
                            #[cfg(target_os = "linux")]
                            let is_wayland = gpui::guess_compositor() == "Wayland";
                            #[cfg(not(target_os = "linux"))]
                            let is_wayland = false;

                            #[cfg(target_os = "linux")]
                            {
                                if is_wayland {
                                    room.share_screen_wayland(cx).detach_and_log_err(cx);
                                }
                            }
                            if !is_wayland {
                                let sources = cx.screen_capture_sources();

                                cx.spawn(async move |room, cx| {
                                    let sources = sources.await??;
                                    let first = sources.into_iter().next();
                                    if let Some(first) = first {
                                        room.update(cx, |room, cx| room.share_screen(first, cx))?
                                            .await
                                    } else {
                                        Ok(())
                                    }
                                })
                                .detach_and_log_err(cx);
                            }
                        };
                    });
                });
            }
        });
    })
    .detach();
}

fn notification_window_options(
    screen: Rc<dyn PlatformDisplay>,
    size: Size<Pixels>,
    cx: &App,
) -> WindowOptions {
    let notification_margin_width = px(16.);
    let notification_margin_height = px(-48.);

    let bounds = gpui::Bounds::<Pixels> {
        origin: screen.bounds().top_right()
            - point(
                size.width + notification_margin_width,
                notification_margin_height,
            ),
        size,
    };

    let app_id = ReleaseChannel::global(cx).app_id();

    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        titlebar: None,
        focus: false,
        show: true,
        kind: WindowKind::PopUp,
        is_movable: false,
        display_id: Some(screen.id()),
        window_background: WindowBackgroundAppearance::Transparent,
        app_id: Some(app_id.to_owned()),
        window_min_size: None,
        window_decorations: Some(WindowDecorations::Client),
        tabbing_identifier: None,
        ..Default::default()
    }
}
