use std::{collections::BTreeMap, error::Error, fmt};

use chrono::{DateTime, Utc};
use collaboration_domain::{
    AggregateId, CommunityId, ForumError, ForumPost, ForumPostCursor, ForumProjection,
    ForumThreadPage, ForumVoteDirection, ForumVoteSummary, Message, MessageContent,
    MessageLifecycleState, NostrEventId, PrincipalId, ThreadCursor,
};
use gpui::{
    AnyElement, Context, Entity, EventEmitter, IntoElement, ListAlignment, ListState, Render, Role,
    SharedString, Window, list, px,
};
use ui::{Button, ButtonStyle, LabelSize, prelude::*};
use util::ResultExt as _;

use crate::message_timeline::{
    MessageTimeline, MessageTimelineAuthor, MessageTimelineAuthorKind, MessageTimelineContext,
    MessageTimelineEntry, MessageTimelineError, MessageTimelinePage, MessageTimelineReaction,
};

const MAX_PRESENTATIONS: usize = 100_000;
const MAX_PAGE_SIZE: usize = 200;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumAuthorPresentation {
    pub principal_id: PrincipalId,
    pub kind: MessageTimelineAuthorKind,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForumPermissions {
    pub create_post: bool,
    pub comment: bool,
    pub vote: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumPostRow {
    pub event_id: NostrEventId,
    pub author: MessageTimelineAuthor,
    pub content: String,
    pub reply_count: u64,
    pub votes: ForumVoteSummary,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForumCommentVoteRow {
    pub event_id: NostrEventId,
    pub author_label: String,
    pub votes: ForumVoteSummary,
}

pub struct ForumSnapshot {
    community_id: CommunityId,
    channel_id: AggregateId,
    archived: bool,
    viewer_principal_id: PrincipalId,
    presentations: BTreeMap<PrincipalId, ForumAuthorPresentation>,
    posts: Vec<ForumPostRow>,
    page_size: usize,
    next_post_cursor: Option<ForumPostCursor>,
    has_more_posts: bool,
}

impl ForumSnapshot {
    pub fn from_projection(
        projection: &ForumProjection<'_>,
        presentations: impl IntoIterator<Item = ForumAuthorPresentation>,
        requested_page_size: usize,
    ) -> Result<Self, ForumViewError> {
        let presentations = normalize_presentations(presentations)?;
        let page_size = requested_page_size.clamp(1, MAX_PAGE_SIZE);
        let page = projection
            .posts(None, page_size)
            .map_err(ForumViewError::Projection)?;
        let posts = page
            .posts
            .iter()
            .map(|post| project_post(post, &presentations))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            community_id: projection.community_id(),
            channel_id: projection.channel_id(),
            archived: projection.is_archived(),
            viewer_principal_id: projection.viewer_principal_id(),
            presentations,
            posts,
            page_size,
            next_post_cursor: page.next_cursor,
            has_more_posts: page.has_more,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForumComposerMode {
    Post,
    Comment {
        root_event_id: NostrEventId,
        parent_event_id: NostrEventId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForumAction {
    CreatePost {
        community_id: CommunityId,
        channel_id: AggregateId,
        content: MessageContent,
    },
    CreateComment {
        community_id: CommunityId,
        channel_id: AggregateId,
        root_event_id: NostrEventId,
        parent_event_id: NostrEventId,
        content: MessageContent,
    },
    Vote {
        community_id: CommunityId,
        channel_id: AggregateId,
        target_event_id: NostrEventId,
        direction: ForumVoteDirection,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForumViewEvent {
    OpenPost(NostrEventId),
    LoadMorePosts,
    LoadMoreComments(NostrEventId),
    Submit(ForumAction),
}

pub struct ForumView {
    community_id: CommunityId,
    channel_id: AggregateId,
    archived: bool,
    viewer_principal_id: PrincipalId,
    permissions: ForumPermissions,
    page_size: usize,
    presentations: BTreeMap<PrincipalId, ForumAuthorPresentation>,
    posts: Vec<ForumPostRow>,
    post_list_state: ListState,
    next_post_cursor: Option<ForumPostCursor>,
    has_more_posts: bool,
    selected_post: Option<ForumPostRow>,
    message_timeline: Option<Entity<MessageTimeline>>,
    comment_vote_rows: Vec<ForumCommentVoteRow>,
    next_comment_cursor: Option<ThreadCursor>,
    has_more_comments: bool,
    total_comments: u64,
    composer_mode: Option<ForumComposerMode>,
    draft: String,
}

impl ForumView {
    pub fn new(snapshot: ForumSnapshot, permissions: ForumPermissions) -> Self {
        let post_list_state = ListState::new(snapshot.posts.len(), ListAlignment::Top, px(1024.0));
        Self {
            community_id: snapshot.community_id,
            channel_id: snapshot.channel_id,
            archived: snapshot.archived,
            viewer_principal_id: snapshot.viewer_principal_id,
            permissions,
            page_size: snapshot.page_size,
            presentations: snapshot.presentations,
            posts: snapshot.posts,
            post_list_state,
            next_post_cursor: snapshot.next_post_cursor,
            has_more_posts: snapshot.has_more_posts,
            selected_post: None,
            message_timeline: None,
            comment_vote_rows: Vec::new(),
            next_comment_cursor: None,
            has_more_comments: false,
            total_comments: 0,
            composer_mode: None,
            draft: String::new(),
        }
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(&self) -> AggregateId {
        self.channel_id
    }

    pub const fn viewer_principal_id(&self) -> PrincipalId {
        self.viewer_principal_id
    }

    pub const fn is_archived(&self) -> bool {
        self.archived
    }

    pub fn posts(&self) -> &[ForumPostRow] {
        &self.posts
    }

    pub const fn has_more_posts(&self) -> bool {
        self.has_more_posts
    }

    pub const fn total_comments(&self) -> u64 {
        self.total_comments
    }

    pub const fn has_more_comments(&self) -> bool {
        self.has_more_comments
    }

    pub fn selected_post(&self) -> Option<&ForumPostRow> {
        self.selected_post.as_ref()
    }

    pub fn message_timeline(&self) -> Option<Entity<MessageTimeline>> {
        self.message_timeline.clone()
    }

    pub fn comment_vote_rows(&self) -> &[ForumCommentVoteRow] {
        &self.comment_vote_rows
    }

    pub fn composer_mode(&self) -> Option<&ForumComposerMode> {
        self.composer_mode.as_ref()
    }

    pub fn draft(&self) -> &str {
        &self.draft
    }

    pub fn load_more_posts(
        &mut self,
        projection: &ForumProjection<'_>,
        cx: &mut Context<Self>,
    ) -> Result<bool, ForumViewError> {
        self.validate_projection_scope(projection)?;
        if !self.has_more_posts {
            return Ok(false);
        }
        let page = projection
            .posts(self.next_post_cursor, self.page_size)
            .map_err(ForumViewError::Projection)?;
        let mut incoming = page
            .posts
            .iter()
            .map(|post| project_post(post, &self.presentations))
            .collect::<Result<Vec<_>, _>>()?;
        if incoming.iter().any(|row| {
            self.posts
                .iter()
                .any(|existing| existing.event_id == row.event_id)
        }) {
            return Err(ForumViewError::DuplicatePost);
        }
        self.posts.append(&mut incoming);
        self.next_post_cursor = page.next_cursor;
        self.has_more_posts = page.has_more;
        self.post_list_state.reset(self.posts.len());
        cx.notify();
        Ok(true)
    }

    pub fn open_thread(
        &mut self,
        projection: &ForumProjection<'_>,
        root_event_id: NostrEventId,
        cx: &mut Context<Self>,
    ) -> Result<(), ForumViewError> {
        self.validate_projection_scope(projection)?;
        let page = projection
            .thread(root_event_id, None, self.page_size)
            .map_err(ForumViewError::Projection)?;
        let selected_post = project_post(&page.post, &self.presentations)?;
        let comment_vote_rows = project_comment_vote_rows(&page, &self.presentations)?;
        let timeline_page =
            project_thread_page(&page, None, &self.presentations, self.community_id)?;
        let timeline = cx.new(MessageTimeline::new);
        timeline
            .update(cx, |timeline, cx| {
                timeline.apply_history_page(timeline_page, cx)
            })
            .map_err(ForumViewError::Timeline)?;
        self.selected_post = Some(selected_post);
        self.message_timeline = Some(timeline);
        self.comment_vote_rows = comment_vote_rows;
        self.next_comment_cursor = page.next_cursor;
        self.has_more_comments = page.has_more;
        self.total_comments = page.total_comments;
        self.composer_mode = None;
        self.draft.clear();
        cx.notify();
        Ok(())
    }

    pub fn load_more_comments(
        &mut self,
        projection: &ForumProjection<'_>,
        cx: &mut Context<Self>,
    ) -> Result<bool, ForumViewError> {
        self.validate_projection_scope(projection)?;
        if !self.has_more_comments {
            return Ok(false);
        }
        let root_event_id = self
            .selected_post
            .as_ref()
            .map(|post| post.event_id)
            .ok_or(ForumViewError::ThreadNotOpen)?;
        let request_cursor = self.next_comment_cursor;
        let page = projection
            .thread(root_event_id, request_cursor, self.page_size)
            .map_err(ForumViewError::Projection)?;
        let timeline_page = project_thread_page(
            &page,
            request_cursor,
            &self.presentations,
            self.community_id,
        )?;
        let mut comment_vote_rows = project_comment_vote_rows(&page, &self.presentations)?;
        if comment_vote_rows.iter().any(|row| {
            self.comment_vote_rows
                .iter()
                .any(|existing| existing.event_id == row.event_id)
        }) {
            return Err(ForumViewError::DuplicateComment);
        }
        self.message_timeline
            .as_ref()
            .ok_or(ForumViewError::ThreadNotOpen)?
            .update(cx, |timeline, cx| {
                timeline.apply_history_page(timeline_page, cx)
            })
            .map_err(ForumViewError::Timeline)?;
        self.comment_vote_rows.append(&mut comment_vote_rows);
        self.next_comment_cursor = page.next_cursor;
        self.has_more_comments = page.has_more;
        self.total_comments = page.total_comments;
        cx.notify();
        Ok(true)
    }

    pub fn close_thread(&mut self, cx: &mut Context<Self>) {
        self.selected_post = None;
        self.message_timeline = None;
        self.comment_vote_rows.clear();
        self.next_comment_cursor = None;
        self.has_more_comments = false;
        self.total_comments = 0;
        self.composer_mode = None;
        self.draft.clear();
        cx.notify();
    }

    pub fn begin_post(&mut self, cx: &mut Context<Self>) -> Result<(), ForumViewError> {
        self.require_write(self.permissions.create_post)?;
        self.composer_mode = Some(ForumComposerMode::Post);
        self.draft.clear();
        cx.notify();
        Ok(())
    }

    pub fn begin_comment(
        &mut self,
        parent_event_id: NostrEventId,
        cx: &mut Context<Self>,
    ) -> Result<(), ForumViewError> {
        self.require_write(self.permissions.comment)?;
        let root_event_id = self
            .selected_post
            .as_ref()
            .map(|post| post.event_id)
            .ok_or(ForumViewError::ThreadNotOpen)?;
        self.composer_mode = Some(ForumComposerMode::Comment {
            root_event_id,
            parent_event_id,
        });
        self.draft.clear();
        cx.notify();
        Ok(())
    }

    pub fn set_draft(&mut self, draft: impl Into<String>, cx: &mut Context<Self>) {
        self.draft = draft.into();
        cx.notify();
    }

    pub fn cancel_composer(&mut self, cx: &mut Context<Self>) {
        self.composer_mode = None;
        self.draft.clear();
        cx.notify();
    }

    pub fn submit(&mut self, cx: &mut Context<Self>) -> Result<ForumAction, ForumViewError> {
        let mode = self
            .composer_mode
            .clone()
            .ok_or(ForumViewError::ComposerNotOpen)?;
        let draft = self.draft.trim();
        if draft.is_empty() {
            return Err(ForumViewError::EmptyDraft);
        }
        let content = MessageContent::new(draft).map_err(ForumViewError::Message)?;
        let action = match mode {
            ForumComposerMode::Post => {
                self.require_write(self.permissions.create_post)?;
                ForumAction::CreatePost {
                    community_id: self.community_id,
                    channel_id: self.channel_id,
                    content,
                }
            }
            ForumComposerMode::Comment {
                root_event_id,
                parent_event_id,
            } => {
                self.require_write(self.permissions.comment)?;
                ForumAction::CreateComment {
                    community_id: self.community_id,
                    channel_id: self.channel_id,
                    root_event_id,
                    parent_event_id,
                    content,
                }
            }
        };
        self.composer_mode = None;
        self.draft.clear();
        cx.emit(ForumViewEvent::Submit(action.clone()));
        cx.notify();
        Ok(action)
    }

    pub fn vote(
        &mut self,
        target_event_id: NostrEventId,
        direction: ForumVoteDirection,
        cx: &mut Context<Self>,
    ) -> Result<ForumAction, ForumViewError> {
        self.require_write(self.permissions.vote)?;
        if !self
            .posts
            .iter()
            .any(|post| post.event_id == target_event_id)
            && self
                .selected_post
                .as_ref()
                .is_none_or(|post| post.event_id != target_event_id)
            && !self
                .comment_vote_rows
                .iter()
                .any(|comment| comment.event_id == target_event_id)
        {
            return Err(ForumViewError::UnknownVoteTarget);
        }
        let action = ForumAction::Vote {
            community_id: self.community_id,
            channel_id: self.channel_id,
            target_event_id,
            direction,
        };
        cx.emit(ForumViewEvent::Submit(action.clone()));
        Ok(action)
    }

    fn validate_projection_scope(
        &self,
        projection: &ForumProjection<'_>,
    ) -> Result<(), ForumViewError> {
        if projection.community_id() != self.community_id
            || projection.channel_id() != self.channel_id
            || projection.viewer_principal_id() != self.viewer_principal_id
            || projection.is_archived() != self.archived
        {
            return Err(ForumViewError::ScopeMismatch);
        }
        Ok(())
    }

    fn require_write(&self, allowed: bool) -> Result<(), ForumViewError> {
        if self.archived {
            Err(ForumViewError::Archived)
        } else if !allowed {
            Err(ForumViewError::PermissionDenied)
        } else {
            Ok(())
        }
    }

    fn render_post_row(
        &self,
        index: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(post) = self.posts.get(index).cloned() else {
            return div().into_any_element();
        };
        let event_id = post.event_id;
        let event_key = event_id_string(event_id);
        let can_vote = !self.archived && self.permissions.vote;
        v_flex()
            .id(SharedString::from(format!("forum-post-{event_key}")))
            .role(Role::ListItem)
            .aria_label(format!(
                "Forum post by {}. Score {}. {} replies.",
                post.author.label, post.votes.score, post.reply_count
            ))
            .w_full()
            .gap_1()
            .p_3()
            .border_b_1()
            .border_color(cx.theme().colors().border_variant)
            .child(div().text_sm().child(post.author.label))
            .child(div().text_ui(cx).child(post.content))
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new(
                            SharedString::from(format!("forum-upvote-{event_key}")),
                            "Upvote",
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .disabled(!can_vote)
                        .on_click(cx.listener(
                            move |this, _, _window, cx| {
                                this.vote(event_id, ForumVoteDirection::Up, cx).log_err();
                            },
                        )),
                    )
                    .child(div().text_sm().child(post.votes.score.to_string()))
                    .child(
                        Button::new(
                            SharedString::from(format!("forum-downvote-{event_key}")),
                            "Downvote",
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .disabled(!can_vote)
                        .on_click(cx.listener(
                            move |this, _, _window, cx| {
                                this.vote(event_id, ForumVoteDirection::Down, cx).log_err();
                            },
                        )),
                    )
                    .child(
                        Button::new(
                            SharedString::from(format!("forum-open-{event_key}")),
                            format!("{} replies", post.reply_count),
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .on_click(cx.listener(
                            move |_this, _, _window, cx| {
                                cx.emit(ForumViewEvent::OpenPost(event_id));
                            },
                        )),
                    ),
            )
            .into_any_element()
    }

    fn render_composer(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(mode) = &self.composer_mode else {
            return div().into_any_element();
        };
        let label = match mode {
            ForumComposerMode::Post => "New forum post",
            ForumComposerMode::Comment { .. } => "New forum comment",
        };
        v_flex()
            .id("forum-composer")
            .role(Role::Form)
            .aria_label(label)
            .w_full()
            .gap_2()
            .p_3()
            .border_1()
            .border_color(cx.theme().colors().border)
            .child(div().text_sm().child(label))
            .child(
                div()
                    .id("forum-composer-input")
                    .role(Role::TextInput)
                    .aria_label("Forum message content")
                    .min_h_16()
                    .w_full()
                    .p_2()
                    .bg(cx.theme().colors().editor_background)
                    .child(if self.draft.is_empty() {
                        SharedString::from("Write a message…")
                    } else {
                        SharedString::from(self.draft.clone())
                    }),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        Button::new("forum-composer-submit", "Submit")
                            .style(ButtonStyle::Filled)
                            .disabled(self.draft.trim().is_empty())
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.submit(cx).log_err();
                            })),
                    )
                    .child(
                        Button::new("forum-composer-cancel", "Cancel")
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|this, _, _window, cx| this.cancel_composer(cx))),
                    ),
            )
            .into_any_element()
    }

    fn render_comment_vote_controls(&self, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let can_vote = !self.archived && self.permissions.vote;
        self.comment_vote_rows
            .iter()
            .map(|row| {
                let event_id = row.event_id;
                let event_key = event_id_string(event_id);
                h_flex()
                    .id(SharedString::from(format!(
                        "forum-comment-votes-{event_key}"
                    )))
                    .role(Role::Group)
                    .aria_label(format!(
                        "Vote on comment by {}. Score {}.",
                        row.author_label, row.votes.score
                    ))
                    .w_full()
                    .gap_1()
                    .px_3()
                    .py_1()
                    .child(
                        Button::new(
                            SharedString::from(format!("forum-comment-upvote-{event_key}")),
                            "Upvote comment",
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .disabled(!can_vote)
                        .on_click(cx.listener(
                            move |this, _, _window, cx| {
                                this.vote(event_id, ForumVoteDirection::Up, cx).log_err();
                            },
                        )),
                    )
                    .child(div().text_sm().child(row.votes.score.to_string()))
                    .child(
                        Button::new(
                            SharedString::from(format!("forum-comment-downvote-{event_key}")),
                            "Downvote comment",
                        )
                        .style(ButtonStyle::Subtle)
                        .label_size(LabelSize::Small)
                        .disabled(!can_vote)
                        .on_click(cx.listener(
                            move |this, _, _window, cx| {
                                this.vote(event_id, ForumVoteDirection::Down, cx).log_err();
                            },
                        )),
                    )
                    .into_any_element()
            })
            .collect()
    }
}

impl EventEmitter<ForumViewEvent> for ForumView {}

impl Render for ForumView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        if let Some(post) = self.selected_post.clone() {
            let event_id = post.event_id;
            return v_flex()
                .id("forum-thread-detail")
                .size_full()
                .child(
                    h_flex()
                        .w_full()
                        .gap_2()
                        .p_2()
                        .border_b_1()
                        .border_color(cx.theme().colors().border)
                        .child(
                            Button::new("forum-back", "Back")
                                .style(ButtonStyle::Subtle)
                                .on_click(
                                    cx.listener(|this, _, _window, cx| this.close_thread(cx)),
                                ),
                        )
                        .child(
                            div()
                                .text_sm()
                                .child(format!("{} comments", self.total_comments)),
                        ),
                )
                .when(self.archived, |this| this.child(archived_banner(cx)))
                .child(
                    v_flex()
                        .w_full()
                        .gap_1()
                        .p_3()
                        .child(div().text_sm().child(post.author.label))
                        .child(div().text_ui(cx).child(post.content)),
                )
                .child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .when_some(self.message_timeline.clone(), |this, timeline| {
                            this.child(timeline)
                        }),
                )
                .children(self.render_comment_vote_controls(cx))
                .when(self.has_more_comments, |this| {
                    this.child(
                        div().p_2().child(
                            Button::new("forum-comments-load-more", "Load more comments")
                                .style(ButtonStyle::Subtle)
                                .on_click(cx.listener(move |_this, _, _window, cx| {
                                    cx.emit(ForumViewEvent::LoadMoreComments(event_id));
                                })),
                        ),
                    )
                })
                .when(!self.archived && self.permissions.comment, |this| {
                    this.child(
                        div().p_2().child(
                            Button::new("forum-reply", "Reply")
                                .style(ButtonStyle::Subtle)
                                .on_click(cx.listener(move |this, _, _window, cx| {
                                    this.begin_comment(event_id, cx).log_err();
                                })),
                        ),
                    )
                })
                .when(self.composer_mode.is_some(), |this| {
                    this.child(self.render_composer(cx))
                });
        }

        let content = if self.posts.is_empty() {
            div()
                .id("forum-empty")
                .role(Role::Status)
                .aria_label("No forum posts")
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child("No posts yet")
                .into_any_element()
        } else {
            list(self.post_list_state.clone(), move |index, window, cx| {
                view.update(cx, |this, cx| this.render_post_row(index, window, cx))
            })
            .size_full()
            .into_any_element()
        };
        v_flex()
            .id("forum-post-list")
            .size_full()
            .when(self.archived, |this| this.child(archived_banner(cx)))
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .p_2()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(div().text_ui(cx).child("Forum"))
                    .child(
                        Button::new("forum-new-post", "New post")
                            .style(ButtonStyle::Filled)
                            .disabled(self.archived || !self.permissions.create_post)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.begin_post(cx).log_err();
                            })),
                    ),
            )
            .child(div().flex_1().min_h_0().child(content))
            .when(self.has_more_posts, |this| {
                this.child(
                    div().p_2().child(
                        Button::new("forum-posts-load-more", "Load more posts")
                            .style(ButtonStyle::Subtle)
                            .on_click(cx.listener(|_this, _, _window, cx| {
                                cx.emit(ForumViewEvent::LoadMorePosts);
                            })),
                    ),
                )
            })
            .when(self.composer_mode.is_some(), |this| {
                this.child(self.render_composer(cx))
            })
    }
}

fn archived_banner(cx: &mut Context<ForumView>) -> impl IntoElement {
    div()
        .id("forum-archived")
        .role(Role::Status)
        .aria_label("Archived forum is read only")
        .w_full()
        .px_3()
        .py_1()
        .bg(cx.theme().status().warning_background.opacity(0.2))
        .text_sm()
        .child("Archived forum · read only")
}

fn normalize_presentations(
    presentations: impl IntoIterator<Item = ForumAuthorPresentation>,
) -> Result<BTreeMap<PrincipalId, ForumAuthorPresentation>, ForumViewError> {
    let mut normalized = BTreeMap::new();
    for (index, presentation) in presentations.into_iter().enumerate() {
        if index >= MAX_PRESENTATIONS {
            return Err(ForumViewError::TooManyPresentations);
        }
        if presentation.label.trim().is_empty() {
            return Err(ForumViewError::EmptyPresentation);
        }
        match normalized.entry(presentation.principal_id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(presentation);
            }
            std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &presentation => {
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                return Err(ForumViewError::ConflictingPresentation);
            }
        }
    }
    Ok(normalized)
}

fn project_post(
    post: &ForumPost<'_>,
    presentations: &BTreeMap<PrincipalId, ForumAuthorPresentation>,
) -> Result<ForumPostRow, ForumViewError> {
    let fields = post.message.fields();
    Ok(ForumPostRow {
        event_id: fields.source.event_id,
        author: project_author(fields.author.principal_id(), presentations)?,
        content: post
            .message
            .visible_content()
            .map_or_else(String::new, |content| content.as_str().to_owned()),
        reply_count: post
            .thread_summary
            .as_ref()
            .map_or(0, |summary| summary.descendant_count),
        votes: post.votes,
        created_at: project_timestamp(fields.source.event_created_at)?,
    })
}

fn project_thread_page(
    page: &ForumThreadPage<'_>,
    request_cursor: Option<ThreadCursor>,
    presentations: &BTreeMap<PrincipalId, ForumAuthorPresentation>,
    community_id: CommunityId,
) -> Result<MessageTimelinePage, ForumViewError> {
    let root_event_id = page.post.message.fields().source.event_id;
    Ok(MessageTimelinePage {
        request_cursor: request_cursor.map(thread_cursor_string),
        next_cursor: page.next_cursor.map(thread_cursor_string),
        entries: page
            .comments
            .iter()
            .map(|comment| {
                project_message(
                    comment.message,
                    Some(comment.parent_event_id),
                    comment.votes,
                    presentations,
                    community_id,
                    root_event_id,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn project_comment_vote_rows(
    page: &ForumThreadPage<'_>,
    presentations: &BTreeMap<PrincipalId, ForumAuthorPresentation>,
) -> Result<Vec<ForumCommentVoteRow>, ForumViewError> {
    page.comments
        .iter()
        .map(|comment| {
            let author = project_author(
                comment.message.fields().author.principal_id(),
                presentations,
            )?;
            Ok(ForumCommentVoteRow {
                event_id: comment.message.fields().source.event_id,
                author_label: author.label,
                votes: comment.votes,
            })
        })
        .collect()
}

fn project_message(
    message: &Message,
    reply_to: Option<NostrEventId>,
    votes: ForumVoteSummary,
    presentations: &BTreeMap<PrincipalId, ForumAuthorPresentation>,
    community_id: CommunityId,
    root_event_id: NostrEventId,
) -> Result<MessageTimelineEntry, ForumViewError> {
    let fields = message.fields();
    let occurred_at = project_timestamp(fields.source.event_created_at)?;
    let mut reactions = Vec::new();
    if votes.upvotes > 0 {
        reactions.push(MessageTimelineReaction {
            value: "Upvotes".into(),
            count: u32::try_from(votes.upvotes).map_err(|_| ForumViewError::VoteCountOverflow)?,
        });
    }
    if votes.downvotes > 0 {
        reactions.push(MessageTimelineReaction {
            value: "Downvotes".into(),
            count: u32::try_from(votes.downvotes).map_err(|_| ForumViewError::VoteCountOverflow)?,
        });
    }
    Ok(MessageTimelineEntry {
        event_id: event_id_string(fields.source.event_id),
        operation_id: None,
        source_version: fields.version.get(),
        author: project_author(fields.author.principal_id(), presentations)?,
        content: message
            .visible_content()
            .map_or_else(String::new, |content| content.as_str().to_owned()),
        reply_to: reply_to.map(event_id_string),
        edited: fields.lifecycle_state == MessageLifecycleState::Edited,
        deleted: fields.lifecycle_state == MessageLifecycleState::Deleted,
        reactions,
        occurred_at,
        projected_at: occurred_at,
        context: MessageTimelineContext {
            community_id: Some(community_id.to_string()),
            project_id: None,
            thread_id: Some(event_id_string(root_event_id)),
        },
    })
}

fn project_author(
    principal_id: PrincipalId,
    presentations: &BTreeMap<PrincipalId, ForumAuthorPresentation>,
) -> Result<MessageTimelineAuthor, ForumViewError> {
    let presentation = presentations
        .get(&principal_id)
        .ok_or(ForumViewError::MissingPresentation(principal_id))?;
    Ok(MessageTimelineAuthor {
        kind: presentation.kind,
        id: principal_id.to_string(),
        label: presentation.label.clone(),
    })
}

fn project_timestamp(timestamp: u64) -> Result<DateTime<Utc>, ForumViewError> {
    let timestamp = i64::try_from(timestamp).map_err(|_| ForumViewError::InvalidTimestamp)?;
    DateTime::from_timestamp(timestamp, 0).ok_or(ForumViewError::InvalidTimestamp)
}

fn event_id_string(event_id: NostrEventId) -> String {
    event_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn thread_cursor_string(cursor: ThreadCursor) -> String {
    format!("{}:{}", cursor.created_at, event_id_string(cursor.event_id))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForumViewError {
    Projection(ForumError),
    Timeline(MessageTimelineError),
    Message(collaboration_domain::MessageError),
    MissingPresentation(PrincipalId),
    EmptyPresentation,
    ConflictingPresentation,
    TooManyPresentations,
    InvalidTimestamp,
    VoteCountOverflow,
    ScopeMismatch,
    DuplicatePost,
    DuplicateComment,
    ThreadNotOpen,
    ComposerNotOpen,
    EmptyDraft,
    Archived,
    PermissionDenied,
    UnknownVoteTarget,
}

impl fmt::Display for ForumViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Projection(error) => error.fmt(formatter),
            Self::Timeline(error) => error.fmt(formatter),
            Self::Message(error) => error.fmt(formatter),
            Self::MissingPresentation(principal_id) => {
                write!(formatter, "forum author {principal_id} has no presentation")
            }
            Self::EmptyPresentation => formatter.write_str("forum author label must not be empty"),
            Self::ConflictingPresentation => {
                formatter.write_str("forum author has conflicting presentations")
            }
            Self::TooManyPresentations => formatter.write_str("too many forum presentations"),
            Self::InvalidTimestamp => formatter.write_str("forum timestamp is out of range"),
            Self::VoteCountOverflow => formatter.write_str("forum vote count is out of range"),
            Self::ScopeMismatch => {
                formatter.write_str("forum projection scope does not match view")
            }
            Self::DuplicatePost => formatter.write_str("forum page repeats a post"),
            Self::DuplicateComment => formatter.write_str("forum page repeats a comment"),
            Self::ThreadNotOpen => formatter.write_str("forum thread is not open"),
            Self::ComposerNotOpen => formatter.write_str("forum composer is not open"),
            Self::EmptyDraft => formatter.write_str("forum draft must not be empty"),
            Self::Archived => formatter.write_str("archived forum is read only"),
            Self::PermissionDenied => formatter.write_str("forum action is not permitted"),
            Self::UnknownVoteTarget => formatter.write_str("forum vote target is not visible"),
        }
    }
}

impl Error for ForumViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Projection(error) => Some(error),
            Self::Timeline(error) => Some(error),
            Self::Message(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{
        AggregateVersion, AuthenticatedPrincipal, AuthorizationAction, AuthorizationRequest,
        AuthorizationResource, AuthorizationResourceKind, AuthorizationScope, Channel,
        ChannelLifecycleState, ChannelMembership, ChannelName, ChannelRecordFields, ChannelType,
        ChannelVisibility, CommunityMembership, ForumMessageInput, MembershipRole,
        MembershipStatus, MessageAuthor, MessageRecordFields, MessageSource, NostrPublicKey,
        PrincipalScopes, ServiceAccountId, TenantContext, ThreadReference, TrustedTenantRoute,
    };
    use gpui::TestAppContext;
    use uuid::Uuid;

    use super::*;

    fn community_id() -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(1))
    }

    fn aggregate_id(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn source(value: u8, created_at: u64) -> MessageSource {
        MessageSource {
            event_id: event_id(value),
            event_created_at: created_at,
        }
    }

    fn channel(lifecycle_state: ChannelLifecycleState) -> Channel {
        Channel::from_record(ChannelRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            name: ChannelName::new("discussions").expect("channel name"),
            channel_type: ChannelType::Forum,
            visibility: ChannelVisibility::Open,
            lifecycle_state,
            description: None,
            creator_principal_id: principal_id(3),
            expiration: None,
            version: AggregateVersion::FIRST,
        })
        .expect("forum channel")
    }

    fn tenant() -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id(), "forum-ui-test")
                    .expect("tenant route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn scope() -> AuthorizationScope {
        AuthorizationScope::new("forum:read").expect("authorization scope")
    }

    fn principal(scope: &AuthorizationScope) -> AuthenticatedPrincipal {
        AuthenticatedPrincipal::zed_account(
            principal_id(20),
            community_id(),
            ServiceAccountId::new(20),
            PrincipalScopes::new([scope.clone()]).expect("principal scopes"),
        )
    }

    fn read_request<'a>(
        tenant: &'a TenantContext,
        principal: &'a AuthenticatedPrincipal,
        scope: &'a AuthorizationScope,
    ) -> AuthorizationRequest<'a> {
        AuthorizationRequest {
            tenant,
            principal,
            required_scope: scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id: community_id(),
                kind: AuthorizationResourceKind::Channel,
                resource_id: aggregate_id(2),
                owner_principal_id: None,
                channel_id: Some(aggregate_id(2)),
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(CommunityMembership {
                community_id: community_id(),
                principal_id: principal.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            current_channel_membership_version: Some(AggregateVersion::FIRST),
            channel_membership: Some(ChannelMembership {
                community_id: community_id(),
                channel_id: aggregate_id(2),
                principal_id: principal.principal_id(),
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            }),
            delegation: None,
            now_millis: 100,
        }
    }

    fn message(message_value: u128, event_value: u8, created_at: u64) -> Message {
        Message::from_record(MessageRecordFields {
            community_id: community_id(),
            channel_id: aggregate_id(2),
            message_id: aggregate_id(message_value),
            author: MessageAuthor::principal(principal_id(10 + message_value)),
            content: MessageContent::new(format!("forum message {message_value}"))
                .expect("message content"),
            lifecycle_state: MessageLifecycleState::Active,
            source: source(event_value, created_at),
            current_source: source(event_value, created_at),
            mutations: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("message")
    }

    fn input(message: &Message, reference: ThreadReference) -> ForumMessageInput<'_> {
        ForumMessageInput {
            message,
            author_public_key: NostrPublicKey::from_bytes(
                [message.fields().source.event_id.as_bytes()[0]; 32],
            ),
            reference,
        }
    }

    fn presentation(message: &Message, label: &str) -> ForumAuthorPresentation {
        ForumAuthorPresentation {
            principal_id: message.fields().author.principal_id(),
            kind: MessageTimelineAuthorKind::Human,
            label: label.into(),
        }
    }

    fn permissions() -> ForumPermissions {
        ForumPermissions {
            create_post: true,
            comment: true,
            vote: true,
        }
    }

    #[gpui::test]
    fn forum_creates_posts_and_comments_from_canonical_thread_detail(cx: &mut TestAppContext) {
        let channel = channel(ChannelLifecycleState::Active);
        let root = message(100, 1, 10);
        let comment = message(101, 2, 20);
        let tenant = tenant();
        let scope = scope();
        let principal = principal(&scope);
        let request = read_request(&tenant, &principal, &scope);
        let projection = ForumProjection::build(
            &channel,
            &request,
            [
                input(&root, ThreadReference::TopLevel),
                input(
                    &comment,
                    ThreadReference::Reply {
                        parent_event_id: event_id(1),
                        root_event_id: Some(event_id(1)),
                    },
                ),
            ],
            [],
        )
        .expect("projection");
        let snapshot = ForumSnapshot::from_projection(
            &projection,
            [presentation(&root, "Avery"), presentation(&comment, "Sam")],
            20,
        )
        .expect("snapshot");
        let view = cx.new(|_| ForumView::new(snapshot, permissions()));

        view.update(cx, ForumView::begin_post)
            .expect("post composer");
        view.update(cx, |view, cx| view.set_draft("  New topic  ", cx));
        let post = view.update(cx, ForumView::submit).expect("post action");
        assert!(matches!(
            post,
            ForumAction::CreatePost {
                community_id: id,
                channel_id,
                content,
            } if id == community_id()
                && channel_id == aggregate_id(2)
                && content.as_str() == "New topic"
        ));

        view.update(cx, |view, cx| {
            view.open_thread(&projection, event_id(1), cx)
        })
        .expect("thread detail");
        let timeline_event_ids = view.read_with(cx, |view, cx| {
            view.message_timeline()
                .expect("timeline")
                .read(cx)
                .timeline()
                .read(cx)
                .items()
                .iter()
                .map(|item| item.id.source_id().to_owned())
                .collect::<Vec<_>>()
        });
        assert_eq!(timeline_event_ids, [event_id_string(event_id(2))]);
        assert_eq!(view.read_with(cx, |view, _| view.total_comments()), 1);
        let comment_vote = view
            .update(cx, |view, cx| {
                view.vote(event_id(2), ForumVoteDirection::Down, cx)
            })
            .expect("comment vote action");
        assert!(matches!(
            comment_vote,
            ForumAction::Vote {
                target_event_id,
                direction: ForumVoteDirection::Down,
                ..
            } if target_event_id == event_id(2)
        ));

        view.update(cx, |view, cx| view.begin_comment(event_id(2), cx))
            .expect("comment composer");
        view.update(cx, |view, cx| view.set_draft("Follow-up", cx));
        let comment_action = view.update(cx, ForumView::submit).expect("comment action");
        assert!(matches!(
            comment_action,
            ForumAction::CreateComment {
                root_event_id,
                parent_event_id,
                content,
                ..
            } if root_event_id == event_id(1)
                && parent_event_id == event_id(2)
                && content.as_str() == "Follow-up"
        ));
    }

    #[gpui::test]
    fn forum_emits_votes_and_denies_unprojected_permissions(cx: &mut TestAppContext) {
        let channel = channel(ChannelLifecycleState::Active);
        let root = message(100, 1, 10);
        let tenant = tenant();
        let scope = scope();
        let principal = principal(&scope);
        let request = read_request(&tenant, &principal, &scope);
        let projection = ForumProjection::build(
            &channel,
            &request,
            [input(&root, ThreadReference::TopLevel)],
            [],
        )
        .expect("projection");
        let snapshot =
            ForumSnapshot::from_projection(&projection, [presentation(&root, "Avery")], 20)
                .expect("snapshot");
        let view = cx.new(|_| {
            ForumView::new(
                snapshot,
                ForumPermissions {
                    vote: true,
                    ..ForumPermissions::default()
                },
            )
        });

        let action = view
            .update(cx, |view, cx| {
                view.vote(event_id(1), ForumVoteDirection::Up, cx)
            })
            .expect("vote action");
        assert!(matches!(
            action,
            ForumAction::Vote {
                target_event_id,
                direction: ForumVoteDirection::Up,
                ..
            } if target_event_id == event_id(1)
        ));
        assert_eq!(
            view.update(cx, ForumView::begin_post),
            Err(ForumViewError::PermissionDenied)
        );
        view.update(cx, |view, cx| {
            view.open_thread(&projection, event_id(1), cx)
        })
        .expect("thread detail");
        assert_eq!(
            view.update(cx, |view, cx| view.begin_comment(event_id(1), cx)),
            Err(ForumViewError::PermissionDenied)
        );
    }

    #[gpui::test]
    fn archived_forum_remains_readable_and_rejects_every_write(cx: &mut TestAppContext) {
        let channel = channel(ChannelLifecycleState::Archived);
        let root = message(100, 1, 10);
        let tenant = tenant();
        let scope = scope();
        let principal = principal(&scope);
        let request = read_request(&tenant, &principal, &scope);
        let projection = ForumProjection::build(
            &channel,
            &request,
            [input(&root, ThreadReference::TopLevel)],
            [],
        )
        .expect("archived projection");
        let snapshot =
            ForumSnapshot::from_projection(&projection, [presentation(&root, "Avery")], 20)
                .expect("snapshot");
        let view = cx.new(|_| ForumView::new(snapshot, permissions()));

        assert!(view.read_with(cx, |view, _| view.is_archived()));
        assert_eq!(view.read_with(cx, |view, _| view.posts().len()), 1);
        view.update(cx, |view, cx| {
            view.open_thread(&projection, event_id(1), cx)
        })
        .expect("archived thread remains readable");
        assert_eq!(
            view.update(cx, ForumView::begin_post),
            Err(ForumViewError::Archived)
        );
        assert_eq!(
            view.update(cx, |view, cx| view.begin_comment(event_id(1), cx)),
            Err(ForumViewError::Archived)
        );
        assert_eq!(
            view.update(cx, |view, cx| {
                view.vote(event_id(1), ForumVoteDirection::Down, cx)
            }),
            Err(ForumViewError::Archived)
        );
    }
}
