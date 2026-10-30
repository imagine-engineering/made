# Collaborative Workspace Reuse and Ownership Matrix

## Inline shortcut and test-capability ownership — Source141

| Surface or capability | Canonical owner | Registration and required proof |
| --- | --- | --- |
| Multiplayer inline-terminal keyboard entry | AgentUI `bind_keys`, application `reload_keymaps`, existing AgentPanel/Project/TerminalView | Restore the scoped binding after defaults and before user overlays; preserve disabled/defaultless/Standard guards. Actual mounted keyboard, no dock, Unicode position, real PTY focus, targeted Stop and matched native proof remain required. |
| Terminal input test instrumentation | Terminal `test-support`, forwarded by TerminalView `test-support` | Expose the existing input log to consuming application tests; no production substitute, new manager or weakened assertion. Normal Copper builds must retain production features. |

<!-- impl: crates/agent_ui/src/agent_ui.rs#bind_keys -->
<!-- impl: crates/zed/src/zed.rs#reload_keymaps -->
<!-- impl: crates/terminal_view/Cargo.toml#[features] -->

## Classification

- **Native**: the Collaborative surface directly hosts an existing Zed entity or action.
- **Adapter**: a registration, routing, identity-check, focus or selection adapter that retains canonical handles but no domain rows.
- **View**: ephemeral render props or view-local layout, focus, scroll, disclosure or optimistic reconciliation state.
- **New**: canonical functionality for which Standard Zed has no current owner.
- **Removed**: duplicate ownership or behavior deleted by the reuse audit.

Rows group only types that have the same capability, owner, lifecycle, registration
and proof. Every Collaborative Workspace type in the audited paths is named in a
row. Test-only helper types elsewhere in `visual_test_runner.rs` are outside the
Collaborative Workspace implementation and are not classified as product owners.

## Workspace composition

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeWorkspace` | Collaborative layout composition | The existing workspace `Project`, timeline/composer/review `AnyView` entities and native status/title bars | Adapter | `Workspace::new` creates one presentation entity behind `multiplayer-tools` | workspace switch/restart and cross-surface tests assert the same `Project` entity |
| `CollaborativeLayout`, `CollaborativeLayoutGeometry`, `DraggedCollaborativeReview` | Rail/timeline/review geometry and drag state | GPUI layout and native review view | View | owned by `CollaborativeWorkspace` | reference, narrow and wide geometry tests |
| `CollaborativeEditorSurface` | Display files opened from native review actions | Existing `Workspace::render_center`, canonical `PaneGroup`, `Pane` and editor entities | Adapter | `CollaborativeWorkspace` retains a weak workspace view; native item activation selects it in the review region | `collaborative_revision_changes_the_native_diff_contents` checks OpenExcerpts, visibility, resizing and retained entities |
| `CollaborativeEditorTabs` | Place the opened file tabs beside the review heading | Existing active `Pane::render_tab_bar` and its item, selection, close and drag actions | Adapter | Weak workspace view, mounted in the review header only for a single native pane; split panes retain their own bars | Header placement/retained tab tests and earlier native Made interaction pass; checkpoint retains full visual-parity gap |
| `CollaborativeLayoutState`, `CollaborativeDockRequests` | Width, collapse, selected review and explicitly requested native dock mounting | Workspace KVP/settings persistence; native `Dock` retains actual open/active/size state | View | loaded/saved by the existing `WorkspaceId` KVP path; no separate panel authority | restart, geometry and canonical panel-action focus tests |
| `CollaborativeAcpComposer`, `CollaborativeAcpComposerFooter` | Render the native input/interventions and co-compose its controls with workspace status | Exact active `ThreadView` and its `MessageEditor` | View-only adapter | Same thread observation and canonical editor callbacks; provider carries an optional footer view | Rendered send/stop interaction, provider replacement and clipping regression; final owner suites and Light/Dark 64-capture runs pass; remaining density gaps in checkpoint |
| `SingleRowConfigOptions` | Keep multiple native ACP configuration controls in the collaborative footer | Existing `ConfigOptionsView` and its exact `ConfigOptionSelector` entities | View-only adapter | Cached by `ConfigOptionsView`, observed through a weak owner handle, mounted only by the Multiplayer `ThreadView` footer | Source reuses native selector entities without config copies; GPUI width/density/focus and regular-wrapping regressions pass. Boolean, picker-operation and watch-refresh coverage remain pending in `multi-acp-layout-validation.json`. |
| `CollaborativeStatusMenu` | Expose secondary native status items and complete execution details | Existing StatusBar item entities, hide settings and participant provider | View | Existing StatusBar renders the overflow popover in Collaborative mode | Native menu keyboard/action and reference-width renderer checks pass; full visual parity remains partial |
| `CollaborativeComposerProvider`, `CollaborativeComposerRegistration`, `CollaborativeComposerHost`, `CollaborativeComposerSurface`, `CollaborativeComposerRegistrationError`, `CollaborativeComposerActionError`, `ComposerAction` | Route one selected native composer and its actions | ACP `MessageEditor` or channel `ChannelMessageComposer` | Adapter | agent/channel adapters register in `Workspace`; generation token unregisters stale selection | composer GPUI test submits/cancels the native ACP editor; channel selection tests replace only the visible provider |
| `CollaborativeTimelineProvider`, `CollaborativeTimelineRegistration`, `CollaborativeTimelineHost`, `CollaborativeTimelineRegistrationError` | Route one selected native timeline | `ThreadView` entry list or `MessageTimeline` | Adapter | agent/channel adapter registration in `Workspace` | timeline test compares registered entity IDs and proves unregister/switch drops the old view |
| `CollaborativeReviewSlot`, `CollaborativeReviewRegistration`, `CollaborativeReviewHost`, `CollaborativeReviewRegistrationError`, `CollaborativeReviewSelectionError` | Route selected native agent/project diff | `AgentDiffPane` and `ProjectDiff` | Adapter | adapters register an `AnyView` by canonical `Project` identity | review tests compare exact native entity IDs and stale registrations fail closed |
| `CollaborativeReviewSummarySource` | Stale-action token for generic Git activity | Native provider entity ID plus its externally supplied revision | Adapter | supplied only with a native action callback | action-routing tests reject provider/revision mismatch; it contains no files, hunks or Git state |
| `CollaborativeReviewAction`, `CollaborativeReviewActionState`, `CollaborativeReviewActionContext`, `CollaborativeReviewActionRequest`, `ExecutedCollaborativeReviewAction`, `CollaborativeReviewActionError` | Validate generic activity-card action routing | Native Git/agent callback is the only mutator | Adapter | `collab_ui::GitActivityCard` emits and invokes native callbacks | native Git regression proves mutation occurs in repository state; stale/conflict routes do not invoke callback |
| `CollaborativeNavigation`, `CollaborativeNavigationTarget`, `CollaborativeProjectTarget`, `CollaborativeRemoteTarget`, `PersistedCollaborativeNavigation`, `CollaborativeNavigationError`, `CollaborativeEntityLinkError` | Collaborative-only selection/back-forward relationships | Native `Workspace` open/focus actions; KVP stores only target/history presentation | View | `Workspace` owns one navigation value | navigation tests prove actions open canonical project/thread/channel owners and restart restores only target identity |
| `CollaborativeFocusRegion`, `CollaborativeFocusOrder` | Cross-region focus order | Native `FocusHandle`s from rail, `ThreadView` composer, review and `StatusBar` | View | assembled by `CollaborativeWorkspace::focus_region_handles` | keyboard-focus tests compare the active native composer focus handle |
| `CollaborativeAccessibilityContract`, `CollaborativeAnnouncement`, `CollaborativeAnnouncementRole` | Labels and announcements | Existing provider state and GPUI roles | View | rendered by native surfaces | accessibility tests cover labels and failure announcements |
| `CollaborativeShellState`, `CollaborativeShellStatus`, `CollaborativeShellScope`, `CollaborativeShellPhase`, `CollaborativeShellRetryRequested` | View-local loading/partial-failure disclosure | Native provider availability; no domain data | View | owned by `CollaborativeWorkspace` | shell-state retry and failure tests |
| `CollaborativeTopBar`, `CollaborativeTopBarActionAvailability`, `CollaborativeTopBarTestSnapshot` | Project title, participants and native layout/share actions | `Project`, live participant reader, native title-bar/call actions and layout entity | Adapter | composed by `title_bar::TitleBar` | top-bar tests observe project/provider updates and native avatar branch |

## Participants, presence and status

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeParticipantProvider`, `CollaborativeParticipantProviderState`, `CollaborativeParticipantRegistration`, `CollaborativeParticipantHost`, `ParticipantStateReader`, `CollaborativeParticipantProviderError` | Route a live participant projection | Reader resolves active `ThreadView`, `ThreadMetadataStore`, `ActiveCall` room and native users on demand | Adapter | Zed composition registers once per active thread and only notifies on canonical updates | provider test changes its canonical reader without replacing copied rows; selection unregisters stale reader |
| `CollaborativeParticipant`, `CollaborativeParticipantIdentity`, `CollaborativeParticipantPresence`, `CollaborativeParticipantViewData`, `CollaborativeConnectionState`, `CollaborativeExecutionStatus`, `CollaborativeExecutionPhase`, `CollaborativeExecutionLocation` | Transient avatar/presence/execution props | `client::User`, room participant, ACP agent/thread/model and metadata | View | produced on each provider read; never persisted | top-bar/status tests observe authoritative user/thread changes |
| `CollaborativeParticipantAdapter`, `CollaborativeParticipantAdapterError`, `RoomStateReader` | Translate active native agent/call metadata | Weak active `ThreadView`, `ThreadMetadataStore`, `ActiveCall` | Adapter | reconciled from the active `AgentPanel` | entity identity and live-reader ownership tests |
| `CollaborativeParticipantStatus` | ACP-specific status-bar addition | live participant provider; native status bar remains the container; project/branch item composition is still incomplete | View | rendered as an extra native status item | status tests cover live execution/failure/unavailable states |
| `CollaborativeAwarenessStore`, `GlobalCollaborativeAwarenessStore`, `CollaborativeAwarenessTarget`, `AwarenessParticipantId`, `AwarenessPresenceStatus`, `AwarenessPresence`, `AwarenessTyping`, `AwarenessReminder`, `DurableAwarenessUpdate`, `CollaborativeAwarenessUpdate`, `CollaborativeAwarenessConnectionToken`, `CollaborativeAwarenessUpdateOutcome`, `CollaborativeAwarenessDisconnectOutcome`, `CollaborativeAwarenessFreshness`, `CollaborativeAwarenessPresentation`, `AwarenessBadge`, `SourceStatus`, `EphemeralAwareness`, `SourceAwareness`, `StoredDurableAwareness`, `TargetObservation`, `CollaborativeAwarenessError` | Duplicate presence/unread/reminder store | Replaced by `ChannelStore`, `ThreadMetadataStore`, active ACP statuses, project and call entities | Removed | no production registration existed; module deleted | architecture test forbids new Collaborative `Store` owners |
| `CollaborativeStatusProjection`, `CollaborativeRepositoryStatus`, `CollaborativeTaskPhase`, `CollaborativeStatus` | Duplicate project/Git/status reducer | Removed; native `StatusBar` owns the container and ACP status. September 10 audit found project/branch items still absent | Removed | former custom status landmark removed | source invariant and Standard/multiplayer status-bar tests |

## Agent timeline, composer and review

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeTimelineAdapter`, `CollaborativeTimelineAdapterError`, `CollaborativeAcpTimeline` | Mount the canonical conversation, including loading, error and authentication states | Same `ConversationView` and its `ThreadView`, `AgentThread`, `EntryViewState`, Markdown/code/tool/diff/terminal renderers as Agent Panel | Adapter | active `AgentPanel` reconciliation registers `ConversationView`; the connected conversation delegates entries to its native `ThreadView` | native-owner invariant and mounted zed application tests cover pending/ready/clear, failed/stale switches and same-thread authentication; rich ACP GPUI/visual gates remain separate |
| `InlineTerminalView`, `MessageEditorEvent::InlineTerminalHistoryExpanded`, `ThreadView`, `ListState` | Accepted terminal-output disclosure and conversation row anchoring (tasks55.3/56.3/59.3/59.6) | Existing TerminalView content mode/1000-line bound; EntryViewState current-editor identity resolves the row; GPUI List owns ordinary and explicit tail pause policies | View-local disclosure/follow state; no execution or persistence authority | Existing Multiplayer editor event route holds only the current draft/historical owner; observation remeasures that exact row; explicit user scroll clears the hold and ordinary Tail resumes at bottom, while Normal remains Normal | Historical Source136 GPUI30 and Source137 full profiles/ten3038/lint/normal-build passes precede Source138; settled native More/tail/reply and Less/collapse remain bounded positives. Captured Source137 canonical composer-focus duplicate-a11y panic requires Source138 Window retry rollback; current code gates/rebuild/native focus/input/restart/full parity remain pending. Observer control remains optional/unexecuted conditional diagnostic, never a substitute for required positive checks; ShowLess output does not prove future-stream following |
| `ConversationView::discard_draft_prompt`, `AgentPanel`, `draft_prompt_store`, `ThreadMetadataStore` | Explicit draft dismissal without release-flush resurrection (tasks21.2/55.3/56.3/59.3/59.6) | Existing ConversationView task lifecycle, AgentPanel retained/ephemeral/active ownership and canonical draft KVP/metadata stores | Per-view discard state; stored draft authority remains in existing stores | AgentPanel marks matching live owners and cancels pending saves before explicit drop/delete; valid normal quit/reconnect/release saves and metadata-only removal remain unchanged | Historical Source137 full profiles/ten-owner runs pass strengthened Sidebar actual-release accepted/live KVP absence and ordinary persistence; Source138 accessibility code still requires fresh validation and native focus/input/restart, not inherited passing status |
| `Window::transact`, `A11y`, `A11yNodeBuilder`, `List` | Accessible conversation focus through retryable autoscroll prepaint (tasks21.2/55.3/56.3/59.6) | Existing canonical GPUI per-window frame/tree/focus/action ownership | Transaction checkpoint in existing owner; no new registry or accessibility authority | Active prepaint failure restores full provisional tree/maps/listener prefixes; committed changes survive and inactive accessibility takes no checkpoint; strict duplicate assertion remains | Source137 main-thread duplicate-a11y panic captured SHA21db2c64…; Source138 code3 candidate and strengthened real accessible List/nested focus/action tests pending; matched Delta reference available, fresh Copper usability/native input/restart required |
| `CollaborativeTimeline`, `CollaborativeTimelineEvent` | Generic activity stream when no richer native renderer exists | `ActivityReducer`; not used for ACP entries | View | owned by `MessageTimeline` and generic event projections | reducer ordering/update/disclosure tests |
| `CollaborativeActivityCard`, `ActivityCardKind`, `ActivityCardTone`, `ActivityInterventionKind`, `ActivityCardIntervention`, `ActivityCardSource`, `ActivityCardPresentation`, `ActivityCardToggleHandler`, `ActivityCardInterventionHandler` | Truthful generic event fallback | Canonical `ActivityItem` projection | View | rendered only by generic `CollaborativeTimeline` | activity-card exhaustive presentation/action tests |
| `CollaborativeComposerAdapter`, `CollaborativeComposerAdapterError` | Register active ACP composer | Exact `MessageEditor` entity and `MessageEditorEvent` lifecycle | Adapter | active `AgentPanel` reconciliation | GPUI test verifies empty rejection, focus, submit, generating state and cancel on the same thread |
| `CollaborativeAgentReviewAdapter`, `CollaborativeAgentReviewError` | Register agent changes | Exact `AgentDiffPane`, `AcpThread` and action log | Adapter | active thread composition | adapter test compares pane entity and rejects cross-project/stale action log |
| `PersonaDraft`, `TeamMemberDraft`, `TeamDraft`, `ManagedAgentDraft`, `CollaborativeAgentSettingsStatus`, `CollaborativeAgentSettingsEvent`, `CollaborativeAgentSettings`, `TeamDraftError` | Unsaved collaborative agent-settings form | Existing settings/agent configuration owners receive validated submission | View | opened from native agent settings UI | settings tests cover validation, save and failure without production fixtures |

## Rail

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeRail` | New left-rail composition and scroll | child native projections below | View | `Sidebar` mounts behind `multiplayer-tools` | rail GPUI tests cover sections, selection and scroll |
| `CollaborativeNavigationGroup`, `CollaborativeNavigationSourceId`, `CollaborativeNavigationRowId`, `CollaborativeNavigationBadge`, `CollaborativeNavigationRow`, `DuplicateCollaborativeNavigationRow`, `CollaborativeNavigationProjection` | Render-time row identity/props | `Project`, worktree/repository, `ChannelStore`, `ThreadMetadataStore` | View | recomputed while rendering each section | projection tests reject duplicate source IDs; owner update tests rerender rows |
| `CollaborativePinned`, `CollaborativePinnedState`, `RecentNavigationCandidate`, `CollaborativePinnedProjection` | Pinned/recent composition | canonical pinned navigation targets, `WorkspaceDb` recents, channels and thread metadata | View | rail section entity observes each owner | pin/unpin and canonical-owner update tests |
| `CollaborativeProjects`, `CollaborativeProjectSource`, `CollaborativeProjectGroupProjection` | Project/community hierarchy | live `MultiWorkspace` projects/worktrees/repositories and `ChannelStore` | View | rail section observes owner entities | project/channel update and navigation tests |
| `CollaborativeTasks`, `CollaborativeTaskState`, `CollaborativeTaskRow` | Task/thread rows | `ThreadMetadataStore` plus live `AgentPanel` thread status | View | rail section observes metadata and workspace | task status/navigation tests |
| `CollaborativeSubtask` | Child session row projection | Native ACP tool-call subagent session links and retained `ConversationView` | View data; no transcript or session storage | Parent `ThreadMetadataStore` task organization persists disclosure; native conversation owns navigation | `collaborative_subtasks_navigate_and_restore_disclosure`; native populated capture |

## Review

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeProjectReviewAdapter`, `CollaborativeProjectReviewError` | Register project changes | Exact `ProjectDiff`, `Project` and Git store entities | Adapter | Zed composition reuses an open `ProjectDiff` or creates the native item | adapter test compares the exact registered `ProjectDiff` entity |
| `CollaborativeReviewAnchor`, `CollaborativeReviewDiffTarget`, `CollaborativeReviewDiffIndex`, `CollaborativeReviewHunkState`, `CollaborativeReviewDiffResolution`, `CollaborativeReviewStaleReason`, `CollaborativeReviewDiffError`, `CollaborativeReviewDiffSourceIdentity`, `CollaborativeReviewDiffSide`, `CollaborativeReviewDiffTargetId` | Parallel file/hunk/commit/revision index | Replaced by native `ProjectDiff` navigation and diff state | Removed | had no production consumer | source invariant rejects reintroduction; adapter identity test covers native owner |
| `CollaborativeReviewFileSummary`, `CollaborativeReviewSummary`, `CollaborativeReviewSummaryError` | Parallel file list and diff totals | Replaced by native `ProjectDiff` and native status-bar Git owners | Removed | former values were test/activity-only | source invariant and native review regression |

## Hosted channel messaging

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `ChannelMessagingTransport`, `ChannelMessagingHub`, `GlobalChannelMessaging`, `ChannelMessagingViews`, `ActiveChannel`, `PendingOperation`, `MessageTarget`, `ChannelMessageComposer` | Versioned RPC/reconnect/operation routing and composer | Collab/PostgreSQL signed-event/message projection is authoritative; Redis is notification-only | New | opened for an authorized selected `Channel`; selection/release closes subscriptions | PostgreSQL/RPC/replay tests and selection cleanup tests; Epic 51 retains the missing full two-client GUI proof |
| `MessageTimeline`, `MessageTimelineState`, `OptimisticState`, `MessageTimelineAuthorKind`, `MessageTimelineAuthor`, `MessageTimelineReaction`, `MessageTimelineContext`, `MessageTimelineEntry`, `OptimisticMessage`, `MessageTimelinePage`, `MessageTimelineError` | Window rendering and bounded optimistic reconciliation | Canonical server records and stable operation outcomes | New/View | transport feeds pages/live outcomes into the selected entity | pagination, duplicate suppression, optimistic replacement, reconnect and restart tests; no database authority exists in UI |

## Visual evidence

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `CollaborativeVisualVariant`, `CollaborativeRasterMetrics`, `ImageComparison`, `TestResult`, `StubAgentServer` | Exact-size test scenario, native test agent and measurements | Production composition and checked-in references | View/test | visual-test binary only | semantic selectors, region metrics, split/collapse assertions and artifact dimensions |

## Removed files and reduced modules

- Deleted `sidebar::collaborative_awareness`; no production adapter fed it.
- Reduced `workspace::collaborative_review_summary` to the action-source token;
  file lists, selection, navigation targets and diff totals remain native-owned.
- Reduced `git_ui::collaborative_review` to the `ProjectDiff` identity adapter.
- Deleted `workspace::collaborative_status`; the native status bar remains the container.
  The September 10 audit found the project/worktree/branch controls were not
  remounted there; Task 10.3 remains partial. Canonical Git state was preserved.
- Replaced participant `view_data` synchronization across Zed, Workspace,
  Collaborative Workspace and Status Bar with a live canonical state reader.

## Explicitly retained adapters

Timeline, composer, participant and review provider/host/registration types remain
because `workspace` cannot depend on `agent_ui`, `git_ui` or `collab_ui` without
creating dependency cycles. They carry only `Entity`, `WeakEntity`, `AnyView`,
callbacks and generation tokens; selection controls their lifetime. Layout,
focus, navigation and accessibility types remain because those relationships are
specific to the Collaborative presentation and are not authoritative domain data.


### Native task organization and row operations (2026-09-08)

`TaskOrganization` and `TaskProjectAssociation` are genuinely new canonical metadata functionality required by 22.5/22.8. Their only authoring owner is the existing `ThreadMetadataStore`/`ThreadMetadataDb`; the new table lives in that existing SQLite domain. Projects are foreign references using canonical path/remote identity, not new Project/Git/session owners. Sidebar rows project these relationships and unread state.

`TaskRenameModal` and `TaskDetailsModal` retain a row identifier, native metadata snapshot/editor handle and transient validation state. Shared RightClickMenu/ContextMenu own menu lifetime/focus. The row menu delegates native title, share-format, transcript export, archive/restore and provider deletion operations without activating the targeted conversation on opening. Shared focus delivery verifies that the same menu is still deployed before focusing it.

## Native footer remediation after 9195ec5103

| Types | Capability | Authoritative owner / reused native behavior | Class | Production registration | Canonical-owner proof |
| --- | --- | --- | --- | --- | --- |
| `ProjectStatus`, `ProjectTitleContext` | Project/worktree/branch footer controls | Existing TitleBar project resolution, RecentProjects, WorktreePicker and native Git picker | Weak view adapter and ephemeral render inputs | TitleBar workspace hook behind multiplayer-tools | Both native appearance suites verify upward menus and Escape; Standard/Multiplayer profile checks pass |
| `ProjectDiffStatus` | Footer diff totals and review navigation | Existing GitPanel cached totals; registered native ProjectDiff review provider | Weak view adapter; no reducer or persistence | Native GitPanel construction replaces the source of the retained status item | Access/replacement regression passes; native diff click reveals the exact retained ProjectDiff in both appearances |

The normal application status registration now lives in `crates/zed/src/workspace_status.rs` and is called by both production startup and the collaborative visual runner. Its body is unchanged apart from extraction; it creates the same native status entities and actions. It adds no data authority or reducer. See `native-status-extraction.json` and the full-status capture suites.
