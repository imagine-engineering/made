use std::{collections::BTreeMap, error::Error, fmt};

use collaboration_domain::{
    AggregateId, AggregateVersion, CommunityId, MembershipRole, ModerationReport,
    ModerationReportReason, ModerationReportState, ModerationReportTarget, ModerationResolution,
};
use gpui::{AnyElement, Context, EventEmitter, IntoElement, Render, Role, SharedString, Window};
use ui::{Button, ButtonStyle, LabelSize, prelude::*};

const MAX_QUEUE_REPORTS: usize = 200;
const MAX_PRESENTATION_LABEL_BYTES: usize = 256;
const MAX_EVIDENCE_SUMMARY_BYTES: usize = 512;

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationEvidenceSummary(String);

impl ModerationEvidenceSummary {
    pub fn new(value: impl Into<String>) -> Result<Self, ModerationQueueError> {
        let value = value.into();
        let value = value.trim();
        if value.is_empty() || value.len() > MAX_EVIDENCE_SUMMARY_BYTES {
            return Err(ModerationQueueError::InvalidPresentation);
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModerationEvidenceSummary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ModerationEvidenceSummary([REDACTED])")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationReportPresentation {
    pub report_id: AggregateId,
    pub target_label: String,
    pub reporter_label: String,
    pub evidence_summary: Option<ModerationEvidenceSummary>,
}

impl fmt::Debug for ModerationReportPresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModerationReportPresentation")
            .field("report_id", &self.report_id)
            .field("presentation", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationQueueAccess {
    Owner,
    Administrator,
}

impl ModerationQueueAccess {
    pub fn from_role(role: MembershipRole) -> Result<Self, ModerationQueueError> {
        match role {
            MembershipRole::Owner => Ok(Self::Owner),
            MembershipRole::Admin => Ok(Self::Administrator),
            MembershipRole::Member | MembershipRole::Guest | MembershipRole::Bot => {
                Err(ModerationQueueError::PermissionDenied)
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ModerationQueueRow {
    report: ModerationReport,
    presentation: ModerationReportPresentation,
}

impl ModerationQueueRow {
    pub const fn report(&self) -> &ModerationReport {
        &self.report
    }

    pub const fn presentation(&self) -> &ModerationReportPresentation {
        &self.presentation
    }

    pub const fn is_open(&self) -> bool {
        matches!(self.report.fields().state, ModerationReportState::Open)
    }
}

impl fmt::Debug for ModerationQueueRow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModerationQueueRow")
            .field("report", &self.report)
            .field("presentation", &"[REDACTED]")
            .finish()
    }
}

pub struct ModerationQueueSnapshot {
    community_id: CommunityId,
    access: ModerationQueueAccess,
    rows: Vec<ModerationQueueRow>,
}

impl ModerationQueueSnapshot {
    pub fn new(
        community_id: CommunityId,
        role: MembershipRole,
        reports: impl IntoIterator<Item = ModerationReport>,
        presentations: impl IntoIterator<Item = ModerationReportPresentation>,
    ) -> Result<Self, ModerationQueueError> {
        let access = ModerationQueueAccess::from_role(role)?;
        let mut presentations = normalize_presentations(presentations)?;
        let mut rows = Vec::new();
        for report in reports {
            if rows.len() >= MAX_QUEUE_REPORTS {
                return Err(ModerationQueueError::TooManyReports);
            }
            if report.fields().community_id != community_id {
                return Err(ModerationQueueError::TenantMismatch);
            }
            let presentation = presentations
                .remove(&report.fields().report_id)
                .ok_or(ModerationQueueError::MissingPresentation)?;
            rows.push(ModerationQueueRow {
                report,
                presentation,
            });
        }
        if !presentations.is_empty() {
            return Err(ModerationQueueError::UnknownPresentation);
        }
        rows.sort_by(|left, right| {
            right
                .is_open()
                .cmp(&left.is_open())
                .then_with(|| {
                    right
                        .report
                        .fields()
                        .filed_source
                        .occurred_at_millis
                        .cmp(&left.report.fields().filed_source.occurred_at_millis)
                })
                .then_with(|| {
                    left.report
                        .fields()
                        .report_id
                        .as_uuid()
                        .cmp(&right.report.fields().report_id.as_uuid())
                })
        });
        Ok(Self {
            community_id,
            access,
            rows,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationQueueAction {
    Dismiss,
    RemoveContent,
    Ban,
    TimeoutOneHour,
    Escalate,
}

impl ModerationQueueAction {
    pub const fn resolution(self) -> ModerationResolution {
        match self {
            Self::Dismiss => ModerationResolution::Dismissed,
            Self::RemoveContent => ModerationResolution::ContentRemoved,
            Self::Ban => ModerationResolution::Banned,
            Self::TimeoutOneHour => ModerationResolution::TimedOut,
            Self::Escalate => ModerationResolution::Escalated,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Dismiss => "Dismiss",
            Self::RemoveContent => "Remove content",
            Self::Ban => "Ban member",
            Self::TimeoutOneHour => "Time out for 1 hour",
            Self::Escalate => "Escalate",
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::Dismiss => "dismiss",
            Self::RemoveContent => "remove-content",
            Self::Ban => "ban",
            Self::TimeoutOneHour => "timeout",
            Self::Escalate => "escalate",
        }
    }

    const fn is_destructive(self) -> bool {
        matches!(self, Self::RemoveContent | Self::Ban | Self::TimeoutOneHour)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModerationQueueActionRequest {
    pub request_id: u64,
    pub community_id: CommunityId,
    pub report: ModerationReport,
    pub expected_version: AggregateVersion,
    pub action: ModerationQueueAction,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModerationQueueEvent {
    Execute(ModerationQueueActionRequest),
    Refresh,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationQueueServiceError {
    Denied,
    Stale,
    PartialFailure,
    Unavailable,
    InvalidResponse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationQueueNotice {
    ConfirmationRequired(ModerationQueueAction),
    InFlight(ModerationQueueAction),
    Succeeded(ModerationQueueAction),
    Denied,
    Stale,
    PartialFailure,
    Unavailable,
    InvalidResponse,
}

impl ModerationQueueNotice {
    pub const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::ConfirmationRequired(_) => "moderation_confirmation_required",
            Self::InFlight(_) => "moderation_action_in_flight",
            Self::Succeeded(_) => "moderation_action_succeeded",
            Self::Denied => "moderation_action_denied",
            Self::Stale => "moderation_action_stale",
            Self::PartialFailure => "moderation_action_partial_failure",
            Self::Unavailable => "moderation_service_unavailable",
            Self::InvalidResponse => "moderation_service_invalid_response",
        }
    }

    const fn is_failure(self) -> bool {
        matches!(
            self,
            Self::Denied
                | Self::Stale
                | Self::PartialFailure
                | Self::Unavailable
                | Self::InvalidResponse
        )
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ConfirmationRequired(action) => match action {
                ModerationQueueAction::Dismiss => "Confirm dismissing this report.",
                ModerationQueueAction::RemoveContent => {
                    "Confirm removing the reported content and resolving the report."
                }
                ModerationQueueAction::Ban => {
                    "Confirm banning the reported member and resolving the report."
                }
                ModerationQueueAction::TimeoutOneHour => {
                    "Confirm timing out the reported member for one hour and resolving the report."
                }
                ModerationQueueAction::Escalate => "Confirm escalating this report.",
            },
            Self::InFlight(_) => "Moderation action in progress.",
            Self::Succeeded(_) => "Moderation action completed.",
            Self::Denied => "This moderation action is not permitted.",
            Self::Stale => "This report changed. Refresh the queue before trying again.",
            Self::PartialFailure => {
                "The moderation action was only partially completed. Refresh before retrying."
            }
            Self::Unavailable => "Moderation is temporarily unavailable. The queue was retained.",
            Self::InvalidResponse => {
                "Moderation returned an invalid response. The queue was retained."
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModerationQueueError {
    PermissionDenied,
    TenantMismatch,
    TooManyReports,
    InvalidPresentation,
    DuplicatePresentation,
    MissingPresentation,
    UnknownPresentation,
    UnknownReport,
    ActionUnavailable,
    Busy,
    MissingConfirmation,
    StaleReport,
    RequestMismatch,
    InvalidServiceResult,
    RequestIdExhausted,
}

impl ModerationQueueError {
    pub const fn diagnostic_code(self) -> &'static str {
        match self {
            Self::PermissionDenied => "moderation_queue_denied",
            Self::TenantMismatch => "moderation_queue_tenant_mismatch",
            Self::TooManyReports => "moderation_queue_too_many_reports",
            Self::InvalidPresentation => "moderation_queue_invalid_presentation",
            Self::DuplicatePresentation => "moderation_queue_duplicate_presentation",
            Self::MissingPresentation => "moderation_queue_missing_presentation",
            Self::UnknownPresentation => "moderation_queue_unknown_presentation",
            Self::UnknownReport => "moderation_queue_unknown_report",
            Self::ActionUnavailable => "moderation_queue_action_unavailable",
            Self::Busy => "moderation_queue_busy",
            Self::MissingConfirmation => "moderation_queue_missing_confirmation",
            Self::StaleReport => "moderation_queue_stale_report",
            Self::RequestMismatch => "moderation_queue_request_mismatch",
            Self::InvalidServiceResult => "moderation_queue_invalid_service_result",
            Self::RequestIdExhausted => "moderation_queue_request_id_exhausted",
        }
    }
}

impl fmt::Display for ModerationQueueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "moderation queue operation failed ({})",
            self.diagnostic_code()
        )
    }
}

impl Error for ModerationQueueError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingAction {
    report_id: AggregateId,
    expected_version: AggregateVersion,
    action: ModerationQueueAction,
}

pub struct ModerationQueueView {
    community_id: CommunityId,
    access: ModerationQueueAccess,
    rows: Vec<ModerationQueueRow>,
    pending_action: Option<PendingAction>,
    active_request: Option<ModerationQueueActionRequest>,
    next_request_id: u64,
    notice: Option<ModerationQueueNotice>,
}

impl ModerationQueueView {
    pub fn new(snapshot: ModerationQueueSnapshot) -> Self {
        Self {
            community_id: snapshot.community_id,
            access: snapshot.access,
            rows: snapshot.rows,
            pending_action: None,
            active_request: None,
            next_request_id: 1,
            notice: None,
        }
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn access(&self) -> ModerationQueueAccess {
        self.access
    }

    pub fn rows(&self) -> &[ModerationQueueRow] {
        &self.rows
    }

    pub const fn notice(&self) -> Option<ModerationQueueNotice> {
        self.notice
    }

    pub const fn active_request(&self) -> Option<&ModerationQueueActionRequest> {
        self.active_request.as_ref()
    }

    pub fn replace_snapshot(
        &mut self,
        snapshot: ModerationQueueSnapshot,
        cx: &mut Context<Self>,
    ) -> Result<(), ModerationQueueError> {
        if snapshot.community_id != self.community_id {
            return Err(ModerationQueueError::TenantMismatch);
        }
        self.access = snapshot.access;
        self.rows = snapshot.rows;
        self.notice = None;
        cx.notify();
        Ok(())
    }

    pub fn request_action(
        &mut self,
        report_id: AggregateId,
        action: ModerationQueueAction,
        cx: &mut Context<Self>,
    ) -> Result<(), ModerationQueueError> {
        if self.active_request.is_some() {
            return Err(ModerationQueueError::Busy);
        }
        let row = self
            .row(report_id)
            .ok_or(ModerationQueueError::UnknownReport)?;
        if !row.is_open() || !action_is_available(action, row.report.fields().target) {
            self.notice = Some(ModerationQueueNotice::Denied);
            cx.notify();
            return Err(ModerationQueueError::ActionUnavailable);
        }
        self.pending_action = Some(PendingAction {
            report_id,
            expected_version: row.report.fields().version,
            action,
        });
        self.notice = Some(ModerationQueueNotice::ConfirmationRequired(action));
        cx.notify();
        Ok(())
    }

    pub fn cancel_confirmation(&mut self, cx: &mut Context<Self>) {
        self.pending_action = None;
        self.notice = None;
        cx.notify();
    }

    pub fn confirm_action(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<ModerationQueueActionRequest, ModerationQueueError> {
        if self.active_request.is_some() {
            return Err(ModerationQueueError::Busy);
        }
        let pending = self
            .pending_action
            .take()
            .ok_or(ModerationQueueError::MissingConfirmation)?;
        let Some(row) = self.row(pending.report_id) else {
            self.notice = Some(ModerationQueueNotice::Stale);
            cx.notify();
            return Err(ModerationQueueError::StaleReport);
        };
        if !row.is_open() || row.report.fields().version != pending.expected_version {
            self.notice = Some(ModerationQueueNotice::Stale);
            cx.notify();
            return Err(ModerationQueueError::StaleReport);
        }
        let report = row.report.clone();
        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(ModerationQueueError::RequestIdExhausted)?;
        let request = ModerationQueueActionRequest {
            request_id,
            community_id: self.community_id,
            report,
            expected_version: pending.expected_version,
            action: pending.action,
        };
        self.active_request = Some(request.clone());
        self.notice = Some(ModerationQueueNotice::InFlight(pending.action));
        cx.emit(ModerationQueueEvent::Execute(request.clone()));
        cx.notify();
        Ok(request)
    }

    pub fn complete_action(
        &mut self,
        request_id: u64,
        authoritative_report: ModerationReport,
        cx: &mut Context<Self>,
    ) -> Result<(), ModerationQueueError> {
        let active = self
            .active_request
            .clone()
            .ok_or(ModerationQueueError::RequestMismatch)?;
        if active.request_id != request_id {
            return Err(ModerationQueueError::RequestMismatch);
        }
        let fields = authoritative_report.fields();
        let original_fields = active.report.fields();
        let expected_next_version = active
            .expected_version
            .next()
            .ok_or(ModerationQueueError::InvalidServiceResult)?;
        if fields.community_id != self.community_id
            || fields.report_id != original_fields.report_id
            || fields.reporter_principal_id != original_fields.reporter_principal_id
            || fields.target != original_fields.target
            || fields.reason != original_fields.reason
            || fields.private_context != original_fields.private_context
            || fields.filed_source != original_fields.filed_source
            || fields.version != expected_next_version
            || !matches!(
                fields.state,
                ModerationReportState::Resolved(resolution)
                    if resolution.resolution == active.action.resolution()
            )
        {
            self.active_request = None;
            self.notice = Some(ModerationQueueNotice::InvalidResponse);
            cx.notify();
            return Err(ModerationQueueError::InvalidServiceResult);
        }
        let Some(row_index) = self
            .rows
            .iter()
            .position(|row| row.report.fields().report_id == fields.report_id)
        else {
            self.active_request = None;
            self.notice = Some(ModerationQueueNotice::InvalidResponse);
            cx.notify();
            return Err(ModerationQueueError::InvalidServiceResult);
        };
        if !self.rows[row_index].is_open()
            || self.rows[row_index].report.fields().version != active.expected_version
        {
            self.active_request = None;
            self.notice = Some(ModerationQueueNotice::Stale);
            cx.notify();
            return Err(ModerationQueueError::StaleReport);
        }
        let action = active.action;
        self.rows[row_index].report = authoritative_report;
        self.active_request = None;
        self.notice = Some(ModerationQueueNotice::Succeeded(action));
        cx.notify();
        Ok(())
    }

    pub fn fail_action(
        &mut self,
        request_id: u64,
        error: ModerationQueueServiceError,
        cx: &mut Context<Self>,
    ) -> Result<(), ModerationQueueError> {
        let active = self
            .active_request
            .clone()
            .ok_or(ModerationQueueError::RequestMismatch)?;
        if active.request_id != request_id {
            return Err(ModerationQueueError::RequestMismatch);
        }
        self.active_request = None;
        self.notice = Some(match error {
            ModerationQueueServiceError::Denied => ModerationQueueNotice::Denied,
            ModerationQueueServiceError::Stale => ModerationQueueNotice::Stale,
            ModerationQueueServiceError::PartialFailure => ModerationQueueNotice::PartialFailure,
            ModerationQueueServiceError::Unavailable => ModerationQueueNotice::Unavailable,
            ModerationQueueServiceError::InvalidResponse => ModerationQueueNotice::InvalidResponse,
        });
        cx.notify();
        Ok(())
    }

    pub fn request_refresh(&self, cx: &mut Context<Self>) {
        cx.emit(ModerationQueueEvent::Refresh);
    }

    fn row(&self, report_id: AggregateId) -> Option<&ModerationQueueRow> {
        self.rows
            .iter()
            .find(|row| row.report.fields().report_id == report_id)
    }

    fn render_row(&self, index: usize, cx: &mut Context<Self>) -> AnyElement {
        let Some(row) = self.rows.get(index) else {
            return div().into_any_element();
        };
        let fields = row.report.fields();
        let report_id = fields.report_id;
        let target = fields.target;
        let is_busy = self.active_request.is_some();
        let state_label = report_state_label(fields.state);
        let row_label = format!(
            "{} report about {} from {}. {}.",
            report_reason_label(fields.reason),
            row.presentation.target_label,
            row.presentation.reporter_label,
            state_label
        );
        v_flex()
            .id(SharedString::from(format!("moderation-report-{index}")))
            .role(Role::ListItem)
            .aria_label(row_label)
            .w_full()
            .gap_2()
            .p_3()
            .border_b_1()
            .border_color(cx.theme().colors().border_variant)
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .min_w_0()
                            .text_ui(cx)
                            .child(row.presentation.target_label.clone()),
                    )
                    .child(div().text_sm().child(state_label)),
            )
            .child(div().text_sm().child(format!(
                "{} · reported by {}",
                report_reason_label(fields.reason),
                row.presentation.reporter_label
            )))
            .when_some(
                row.presentation.evidence_summary.as_ref(),
                |this, summary| {
                    this.child(
                        div()
                            .id(SharedString::from(format!("moderation-evidence-{index}")))
                            .role(Role::Group)
                            .aria_label("Private moderation evidence summary")
                            .w_full()
                            .p_2()
                            .bg(cx.theme().colors().editor_background)
                            .text_sm()
                            .child(summary.as_str().to_owned()),
                    )
                },
            )
            .when(row.is_open(), |this| {
                this.child(h_flex().w_full().flex_wrap().gap_1().children(
                    available_actions(target).map(|action| {
                        Button::new(
                            SharedString::from(format!("moderation-{index}-{}", action.id())),
                            action.label(),
                        )
                        .style(if action.is_destructive() {
                            ButtonStyle::Tinted(ui::TintColor::Error)
                        } else {
                            ButtonStyle::Subtle
                        })
                        .label_size(LabelSize::Small)
                        .disabled(is_busy)
                        .on_click(cx.listener(
                            move |this, _, _window, cx| {
                                if this.request_action(report_id, action, cx).is_err() {
                                    cx.notify();
                                }
                            },
                        ))
                    }),
                ))
            })
            .into_any_element()
    }
}

impl EventEmitter<ModerationQueueEvent> for ModerationQueueView {}

impl Render for ModerationQueueView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let report_rows = (0..self.rows.len())
            .map(|index| self.render_row(index, cx))
            .collect::<Vec<_>>();
        let content = if self.rows.is_empty() {
            div()
                .id("moderation-queue-empty")
                .role(Role::Status)
                .aria_label("No moderation reports")
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child("No reports to review")
                .into_any_element()
        } else {
            v_flex()
                .id("moderation-queue-reports")
                .role(Role::List)
                .aria_label("Moderation reports")
                .w_full()
                .children(report_rows)
                .into_any_element()
        };

        v_flex()
            .id("moderation-queue")
            .aria_label("Community moderation queue")
            .size_full()
            .min_w_0()
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .gap_2()
                    .p_2()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(div().text_ui(cx).child("Moderation queue"))
                    .child(
                        Button::new("moderation-refresh", "Refresh")
                            .style(ButtonStyle::Subtle)
                            .disabled(self.active_request.is_some())
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.request_refresh(cx);
                            })),
                    ),
            )
            .when_some(self.notice, |this, notice| {
                this.child(
                    v_flex()
                        .id("moderation-notice")
                        .role(if notice.is_failure() {
                            Role::Alert
                        } else {
                            Role::Status
                        })
                        .aria_label(notice.label())
                        .w_full()
                        .gap_1()
                        .px_3()
                        .py_2()
                        .bg(cx.theme().colors().editor_background)
                        .child(notice.label())
                        .when(
                            matches!(notice, ModerationQueueNotice::ConfirmationRequired(_)),
                            |this| {
                                this.child(
                                    h_flex()
                                        .gap_1()
                                        .child(
                                            Button::new("moderation-confirm", "Confirm")
                                                .style(ButtonStyle::Filled)
                                                .on_click(cx.listener(|this, _, _window, cx| {
                                                    if this.confirm_action(cx).is_err() {
                                                        cx.notify();
                                                    }
                                                })),
                                        )
                                        .child(
                                            Button::new("moderation-cancel", "Cancel")
                                                .style(ButtonStyle::Subtle)
                                                .on_click(cx.listener(|this, _, _window, cx| {
                                                    this.cancel_confirmation(cx);
                                                })),
                                        ),
                                )
                            },
                        )
                        .when(notice.is_failure(), |this| {
                            this.child(
                                Button::new("moderation-failure-refresh", "Refresh queue")
                                    .style(ButtonStyle::Subtle)
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.request_refresh(cx);
                                    })),
                            )
                        }),
                )
            })
            .child(
                div()
                    .id("moderation-queue-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(content),
            )
    }
}

fn normalize_presentations(
    presentations: impl IntoIterator<Item = ModerationReportPresentation>,
) -> Result<BTreeMap<AggregateId, ModerationReportPresentation>, ModerationQueueError> {
    let mut normalized = BTreeMap::new();
    for presentation in presentations {
        if presentation.target_label.trim().is_empty()
            || presentation.target_label.len() > MAX_PRESENTATION_LABEL_BYTES
            || presentation.reporter_label.trim().is_empty()
            || presentation.reporter_label.len() > MAX_PRESENTATION_LABEL_BYTES
        {
            return Err(ModerationQueueError::InvalidPresentation);
        }
        if normalized
            .insert(presentation.report_id, presentation)
            .is_some()
        {
            return Err(ModerationQueueError::DuplicatePresentation);
        }
    }
    Ok(normalized)
}

fn action_is_available(action: ModerationQueueAction, target: ModerationReportTarget) -> bool {
    match action {
        ModerationQueueAction::Dismiss | ModerationQueueAction::Escalate => true,
        ModerationQueueAction::RemoveContent => {
            matches!(target, ModerationReportTarget::Event(_))
        }
        ModerationQueueAction::Ban | ModerationQueueAction::TimeoutOneHour => {
            matches!(target, ModerationReportTarget::Principal(_))
        }
    }
}

fn available_actions(
    target: ModerationReportTarget,
) -> impl Iterator<Item = ModerationQueueAction> {
    [
        ModerationQueueAction::Dismiss,
        ModerationQueueAction::RemoveContent,
        ModerationQueueAction::Ban,
        ModerationQueueAction::TimeoutOneHour,
        ModerationQueueAction::Escalate,
    ]
    .into_iter()
    .filter(move |action| action_is_available(*action, target))
}

const fn report_reason_label(reason: ModerationReportReason) -> &'static str {
    match reason {
        ModerationReportReason::Spam => "Spam",
        ModerationReportReason::Profanity => "Profanity",
        ModerationReportReason::IllegalContent => "Illegal content",
        ModerationReportReason::Nudity => "Nudity",
        ModerationReportReason::Malware => "Malware",
        ModerationReportReason::Impersonation => "Impersonation",
        ModerationReportReason::Other => "Other",
    }
}

const fn report_state_label(state: ModerationReportState) -> &'static str {
    match state {
        ModerationReportState::Open => "Open",
        ModerationReportState::Resolved(resolution) => match resolution.resolution {
            ModerationResolution::Dismissed => "Dismissed",
            ModerationResolution::ContentRemoved => "Content removed",
            ModerationResolution::MemberRemoved => "Member removed",
            ModerationResolution::TimedOut => "Timed out",
            ModerationResolution::Banned => "Banned",
            ModerationResolution::Escalated => "Escalated",
        },
    }
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{
        AggregateId, AggregateVersion, CommunityId, MembershipRole, ModerationCommandSource,
        ModerationReport, ModerationReportReason, ModerationReportRecordFields,
        ModerationReportState, ModerationReportTarget, ModerationResolution,
        ModerationResolutionRecord, NostrEventId, OperationId, PrincipalId,
    };
    use gpui::{AppContext as _, TestAppContext};
    use uuid::Uuid;

    use super::{
        ModerationEvidenceSummary, ModerationQueueAction, ModerationQueueError,
        ModerationQueueNotice, ModerationQueueServiceError, ModerationQueueSnapshot,
        ModerationQueueView, ModerationReportPresentation,
    };

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn source(value: u128, occurred_at_millis: u64) -> ModerationCommandSource {
        ModerationCommandSource {
            operation_id: OperationId::from_uuid(Uuid::from_u128(value)),
            occurred_at_millis,
        }
    }

    fn open_report(
        community_id: CommunityId,
        report_value: u128,
        target: ModerationReportTarget,
    ) -> ModerationReport {
        ModerationReport::from_record(ModerationReportRecordFields {
            report_id: aggregate(report_value),
            community_id,
            reporter_principal_id: principal(20 + report_value),
            target,
            reason: ModerationReportReason::Spam,
            private_context: None,
            filed_source: source(
                100 + report_value,
                u64::try_from(report_value).expect("report timestamp"),
            ),
            state: ModerationReportState::Open,
            version: AggregateVersion::FIRST,
        })
        .expect("open report")
    }

    fn resolved_report(
        report: &ModerationReport,
        resolution: ModerationResolution,
    ) -> ModerationReport {
        let fields = report.fields();
        ModerationReport::from_record(ModerationReportRecordFields {
            report_id: fields.report_id,
            community_id: fields.community_id,
            reporter_principal_id: fields.reporter_principal_id,
            target: fields.target,
            reason: fields.reason,
            private_context: fields.private_context.clone(),
            filed_source: fields.filed_source,
            state: ModerationReportState::Resolved(ModerationResolutionRecord {
                resolution,
                actor_principal_id: principal(2),
                source: source(900 + u128::from(fields.version.get()), 10_000),
            }),
            version: fields.version.next().expect("resolved version"),
        })
        .expect("resolved report")
    }

    fn presentation(
        report: &ModerationReport,
        evidence: Option<&str>,
    ) -> ModerationReportPresentation {
        ModerationReportPresentation {
            report_id: report.fields().report_id,
            target_label: "Reported member".to_owned(),
            reporter_label: "Community member".to_owned(),
            evidence_summary: evidence
                .map(|value| ModerationEvidenceSummary::new(value).expect("evidence summary")),
        }
    }

    fn snapshot(
        community_id: CommunityId,
        reports: Vec<ModerationReport>,
    ) -> ModerationQueueSnapshot {
        let presentations = reports
            .iter()
            .map(|report| presentation(report, Some("Private evidence summary")))
            .collect::<Vec<_>>();
        ModerationQueueSnapshot::new(community_id, MembershipRole::Admin, reports, presentations)
            .expect("queue snapshot")
    }

    #[gpui::test]
    fn moderation_queue_exposes_an_accessible_empty_state(cx: &mut TestAppContext) {
        let community_id = community(1);
        let view = cx.new(|_| ModerationQueueView::new(snapshot(community_id, Vec::new())));

        assert!(view.read_with(cx, |view, _| view.rows().is_empty()));
        assert_eq!(view.read_with(cx, |view, _| view.notice()), None);
        assert_eq!(
            ModerationQueueSnapshot::new(
                community_id,
                MembershipRole::Member,
                Vec::new(),
                Vec::new(),
            )
            .err(),
            Some(ModerationQueueError::PermissionDenied)
        );
    }

    #[gpui::test]
    fn moderation_queue_confirms_and_applies_resolution(cx: &mut TestAppContext) {
        let community_id = community(1);
        let report = open_report(
            community_id,
            10,
            ModerationReportTarget::Event(NostrEventId::from_bytes([3; 32])),
        );
        let report_id = report.fields().report_id;
        let view = cx.new(|_| ModerationQueueView::new(snapshot(community_id, vec![report])));

        view.update(cx, |view, cx| {
            view.request_action(report_id, ModerationQueueAction::RemoveContent, cx)
        })
        .expect("confirmation");
        assert_eq!(
            view.read_with(cx, |view, _| view.notice()),
            Some(ModerationQueueNotice::ConfirmationRequired(
                ModerationQueueAction::RemoveContent
            ))
        );
        let request = view
            .update(cx, ModerationQueueView::confirm_action)
            .expect("action request");
        assert_eq!(request.expected_version, AggregateVersion::FIRST);
        let authoritative = view.read_with(cx, |view, _| {
            resolved_report(
                view.rows()[0].report(),
                ModerationResolution::ContentRemoved,
            )
        });
        view.update(cx, |view, cx| {
            view.complete_action(request.request_id, authoritative, cx)
        })
        .expect("action completion");

        assert_eq!(
            view.read_with(cx, |view, _| view.notice()),
            Some(ModerationQueueNotice::Succeeded(
                ModerationQueueAction::RemoveContent
            ))
        );
        assert!(!view.read_with(cx, |view, _| view.rows()[0].is_open()));
    }

    #[gpui::test]
    fn moderation_queue_denies_inapplicable_action(cx: &mut TestAppContext) {
        let community_id = community(1);
        let report = open_report(
            community_id,
            11,
            ModerationReportTarget::Principal(principal(50)),
        );
        let report_id = report.fields().report_id;
        let view = cx.new(|_| ModerationQueueView::new(snapshot(community_id, vec![report])));

        assert_eq!(
            view.update(cx, |view, cx| {
                view.request_action(report_id, ModerationQueueAction::RemoveContent, cx)
            }),
            Err(ModerationQueueError::ActionUnavailable)
        );
        assert_eq!(
            view.read_with(cx, |view, _| view.notice()),
            Some(ModerationQueueNotice::Denied)
        );
        assert!(view.read_with(cx, |view, _| view.active_request().is_none()));
    }

    #[gpui::test]
    fn moderation_queue_rejects_stale_confirmation_after_refresh(cx: &mut TestAppContext) {
        let community_id = community(1);
        let report = open_report(
            community_id,
            12,
            ModerationReportTarget::Principal(principal(51)),
        );
        let report_id = report.fields().report_id;
        let authoritative = resolved_report(&report, ModerationResolution::Dismissed);
        let view = cx.new(|_| ModerationQueueView::new(snapshot(community_id, vec![report])));

        view.update(cx, |view, cx| {
            view.request_action(report_id, ModerationQueueAction::Dismiss, cx)
        })
        .expect("confirmation");
        view.update(cx, |view, cx| {
            view.replace_snapshot(snapshot(community_id, vec![authoritative]), cx)
        })
        .expect("refreshed snapshot");
        assert_eq!(
            view.update(cx, ModerationQueueView::confirm_action),
            Err(ModerationQueueError::StaleReport)
        );
        assert_eq!(
            view.read_with(cx, |view, _| view.notice()),
            Some(ModerationQueueNotice::Stale)
        );
    }

    #[gpui::test]
    fn moderation_queue_retains_report_on_redacted_partial_failure(cx: &mut TestAppContext) {
        let community_id = community(1);
        let secret = "private reporter detail must remain in the operator row only";
        let report = open_report(
            community_id,
            13,
            ModerationReportTarget::Principal(principal(52)),
        );
        let report_id = report.fields().report_id;
        let presentation = presentation(&report, Some(secret));
        let snapshot = ModerationQueueSnapshot::new(
            community_id,
            MembershipRole::Owner,
            vec![report],
            vec![presentation],
        )
        .expect("queue snapshot");
        let view = cx.new(|_| ModerationQueueView::new(snapshot));

        view.update(cx, |view, cx| {
            view.request_action(report_id, ModerationQueueAction::Ban, cx)
        })
        .expect("confirmation");
        let request = view
            .update(cx, ModerationQueueView::confirm_action)
            .expect("action request");
        view.update(cx, |view, cx| {
            view.fail_action(
                request.request_id,
                ModerationQueueServiceError::PartialFailure,
                cx,
            )
        })
        .expect("partial failure");

        let notice = view
            .read_with(cx, |view, _| view.notice())
            .expect("failure notice");
        assert_eq!(notice, ModerationQueueNotice::PartialFailure);
        assert!(!notice.label().contains(secret));
        assert!(view.read_with(cx, |view, _| view.rows()[0].is_open()));
        assert!(view.read_with(cx, |view, _| view.active_request().is_none()));
        let row_debug = view.read_with(cx, |view, _| format!("{:?}", view.rows()[0]));
        assert!(!row_debug.contains(secret));
    }
}
