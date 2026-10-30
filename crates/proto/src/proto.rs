#![allow(non_snake_case)]

pub mod error;
mod macros;
mod typed_envelope;

pub use error::*;
pub use prost::{DecodeError, Message};
use std::{
    cmp,
    fmt::Debug,
    iter, mem,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
pub use typed_envelope::*;

include!(concat!(env!("OUT_DIR"), "/zed.messages.rs"));

pub const REMOTE_SERVER_PEER_ID: PeerId = PeerId { owner_id: 0, id: 0 };
pub const REMOTE_SERVER_PROJECT_ID: u64 = 0;

pub const COLLABORATION_FILE_POLICY_VERSION: u32 = 1;
pub const MAX_COLLABORATION_FILE_POLICY_SCOPES: usize = 64;
pub const MAX_COLLABORATION_FILE_POLICY_PATH_BYTES: usize = 4096;
pub const MAX_COLLABORATION_FILE_POLICY_TOTAL_PATH_BYTES: usize = 64 * 1024;
pub const MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES: usize = 64 * 1024;

fn validate_collaboration_file_policy_scopes(
    scopes: &[CollaborationFilePolicyScope],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !scopes.is_empty() && scopes.len() <= MAX_COLLABORATION_FILE_POLICY_SCOPES,
        "invalid collaboration policy scope count"
    );
    let mut logical_scopes = std::collections::HashSet::new();
    let mut total_path_bytes = 0usize;
    for scope in scopes {
        anyhow::ensure!(
            !scope.abs_path.is_empty(),
            "missing collaboration policy root path"
        );
        anyhow::ensure!(
            matches!(
                CollaborationFilePolicyRootKind::from_i32(scope.root_kind),
                Some(
                    CollaborationFilePolicyRootKind::Directory
                        | CollaborationFilePolicyRootKind::File
                )
            ),
            "unknown collaboration policy root kind"
        );
        anyhow::ensure!(
            matches!(
                scope
                    .path_style
                    .and_then(CollaborationFilePolicyPathStyle::from_i32),
                Some(
                    CollaborationFilePolicyPathStyle::Posix
                        | CollaborationFilePolicyPathStyle::Windows
                )
            ),
            "missing or unknown collaboration policy path style"
        );
        for path in [&scope.abs_path, &scope.root_name, &scope.path] {
            anyhow::ensure!(
                path.len() <= MAX_COLLABORATION_FILE_POLICY_PATH_BYTES,
                "collaboration policy path exceeds byte limit"
            );
            anyhow::ensure!(
                !path.contains('\0'),
                "invalid collaboration policy path encoding"
            );
            total_path_bytes = total_path_bytes
                .checked_add(path.len())
                .ok_or_else(|| anyhow::anyhow!("collaboration policy path byte overflow"))?;
            anyhow::ensure!(
                total_path_bytes <= MAX_COLLABORATION_FILE_POLICY_TOTAL_PATH_BYTES,
                "collaboration policy aggregate path bytes exceed limit"
            );
        }
        anyhow::ensure!(
            logical_scopes.insert((scope.worktree_id, scope.path.as_str())),
            "duplicate collaboration policy scope"
        );
    }
    Ok(())
}

impl GetCollaborationFilePolicy {
    /// These limits bound known-field re-encoding; they cannot bound a frame already decoded.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.protocol_version == COLLABORATION_FILE_POLICY_VERSION,
            "unsupported collaboration policy version"
        );
        validate_collaboration_file_policy_scopes(&self.scopes)?;
        anyhow::ensure!(
            self.encoded_len() <= MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES,
            "collaboration policy encoded payload exceeds limit"
        );
        Ok(())
    }
}

impl GetCollaborationFilePolicyResponse {
    /// A structurally valid Deny response still refuses publication.
    pub fn validate_for_request(&self, request: &GetCollaborationFilePolicy) -> anyhow::Result<()> {
        request.validate()?;
        anyhow::ensure!(
            self.protocol_version == COLLABORATION_FILE_POLICY_VERSION,
            "unsupported collaboration policy version"
        );
        anyhow::ensure!(
            self.project_id == request.project_id,
            "collaboration policy project mismatch"
        );
        anyhow::ensure!(
            self.checked_settings_revision.is_some(),
            "missing checked collaboration policy revision"
        );
        anyhow::ensure!(
            self.results.len() == request.scopes.len(),
            "collaboration policy result cardinality mismatch"
        );
        for (result, expected_scope) in self.results.iter().zip(&request.scopes) {
            let scope = result
                .scope
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing collaboration policy result scope"))?;
            anyhow::ensure!(
                scope == expected_scope,
                "collaboration policy scope echo mismatch"
            );
            anyhow::ensure!(
                matches!(
                    CollaborationFilePolicyDecision::from_i32(result.decision),
                    Some(
                        CollaborationFilePolicyDecision::Allow
                            | CollaborationFilePolicyDecision::Deny
                    )
                ),
                "unknown collaboration policy decision"
            );
        }
        anyhow::ensure!(
            self.encoded_len() <= MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES,
            "collaboration policy encoded payload exceeds limit"
        );
        Ok(())
    }
}

impl CollaborationFilePolicyInvalidated {
    /// Empty roots invalidate the whole project; an absent revision cannot grant permission.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.protocol_version == COLLABORATION_FILE_POLICY_VERSION,
            "unsupported collaboration policy version"
        );
        anyhow::ensure!(
            self.worktree_ids.len() <= MAX_COLLABORATION_FILE_POLICY_SCOPES,
            "invalid collaboration policy invalidation root count"
        );
        let mut roots = std::collections::HashSet::new();
        anyhow::ensure!(
            self.worktree_ids
                .iter()
                .all(|worktree_id| roots.insert(*worktree_id)),
            "duplicate collaboration policy invalidation root"
        );
        anyhow::ensure!(
            self.encoded_len() <= MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES,
            "collaboration policy encoded payload exceeds limit"
        );
        Ok(())
    }
}

messages!(
    (Ack, Foreground),
    (AckBufferOperation, Background),
    (AckChannelMessage, Background),
    (ActivateToolchain, Foreground),
    (ActiveToolchain, Foreground),
    (ActiveToolchainResponse, Foreground),
    (ResolveToolchain, Background),
    (ResolveToolchainResponse, Background),
    (AddNotification, Foreground),
    (AddProjectCollaborator, Foreground),
    (AddWorktree, Foreground),
    (AddWorktreeResponse, Foreground),
    (AllocateWorktreeId, Foreground),
    (AllocateWorktreeIdResponse, Foreground),
    (ApplyCodeAction, Background),
    (ApplyCodeActionResponse, Background),
    (ApplyCompletionAdditionalEdits, Background),
    (ApplyCompletionAdditionalEditsResponse, Background),
    (BlameBuffer, Foreground),
    (BlameBufferResponse, Foreground),
    (BufferReloaded, Foreground),
    (BufferSaved, Foreground),
    (Call, Foreground),
    (CallCanceled, Foreground),
    (CancelCall, Foreground),
    (CancelCargoWorkspace, Foreground),
    (CancelLanguageServerWork, Foreground),
    (ChannelMessageSent, Foreground),
    (ChannelMessageUpdate, Foreground),
    (CollaborativeMessageStreamUpdate, Foreground),
    (OpenCollaborativeChannel, Foreground),
    (OpenCollaborativeChannelResponse, Foreground),
    (CloseCollaborativeChannel, Foreground),
    (GetCollaborativeMessageWindow, Background),
    (GetCollaborativeMessageWindowResponse, Background),
    (ApplyCollaborativeMessageOperation, Foreground),
    (ApplyCollaborativeMessageOperationResponse, Foreground),
    (CloseBuffer, Foreground),
    (Commit, Background),
    (RunGitHook, Background),
    (CopyProjectEntry, Foreground),
    (CreateBufferForPeer, Foreground),
    (CreateImageForPeer, Foreground),
    (CreateFileForPeer, Foreground),
    (CreateChannel, Foreground),
    (CreateChannelResponse, Foreground),
    (CreateProjectEntry, Foreground),
    (CreateRoom, Foreground),
    (CreateRoomResponse, Foreground),
    (DeclineCall, Foreground),
    (DeleteChannel, Foreground),
    (DeleteNotification, Foreground),
    (DeleteProjectEntry, Foreground),
    (TrashProjectEntry, Foreground),
    (TrashProjectEntryResponse, Foreground),
    (RestoreProjectEntry, Foreground),
    (RestoreProjectEntryResponse, Foreground),
    (DownloadFileByPath, Background),
    (DownloadFileResponse, Background),
    (EndStream, Foreground),
    (Error, Foreground),
    (ExpandProjectEntry, Foreground),
    (ExpandProjectEntryResponse, Foreground),
    (FindSearchCandidates, Background),
    (FlushBufferedMessages, Foreground),
    (ExpandAllForProjectEntry, Foreground),
    (ExpandAllForProjectEntryResponse, Foreground),
    (Follow, Foreground),
    (FollowResponse, Foreground),
    (ApplyCodeActionKind, Foreground),
    (ApplyCodeActionKindResponse, Foreground),
    (FormatBuffers, Foreground),
    (FormatBuffersResponse, Foreground),
    (FuzzySearchUsers, Foreground),
    (GetChannelMembers, Foreground),
    (GetChannelMembersResponse, Foreground),
    (GetChannelMessages, Background),
    (GetChannelMessagesById, Background),
    (GetChannelMessagesResponse, Background),
    (GetCargoWorkspace, Background),
    (CargoWorkspaceResponse, Background),
    (GetRustTestDiscovery, Background),
    (CancelRustTestDiscovery, Foreground),
    (RustTestDiscoveryResponse, Background),
    (GetJvmTestDiscovery, Background),
    (CancelJvmTestDiscovery, Foreground),
    (JvmTestDiscoveryResponse, Background),
    (GetJdtlsProjectSnapshot, Background),
    (JdtlsProjectSnapshotResponse, Background),
    (GetKotlinWorkspaceSnapshot, Background),
    (KotlinWorkspaceSnapshotResponse, Background),
    (SetJvmProjectConsumers, Foreground),
    (JvmProjectConsumersResponse, Foreground),
    (ResolveJvmTestAction, Foreground),
    (JvmTestActionPlanResponse, Foreground),
    (UpdateJvmTestRun, Foreground),
    (ResolveRustTestAction, Foreground),
    (RustTestActionPlanResponse, Foreground),
    (UpdateRustTestRun, Foreground),
    (GetStructuredExecutionSnapshot, Background),
    (StructuredExecutionSnapshotPage, Background),
    (GetStructuredExecutionEvents, Background),
    (StructuredExecutionEventChunk, Background),
    (GetSourceCoverage, Background),
    (SourceCoverageResponse, Background),
    (InterpretRustCoverageArtifact, Background),
    (ImportProfileResults, Background),
    (PrepareProfileTaskBinding, Background),
    (GetProfileTaskBinding, Background),
    (CancelProfileTaskBinding, Foreground),
    (ProfileTaskBindingResponse, Background),
    (GetProfileResultPage, Background),
    (CancelProfileImport, Foreground),
    (ProfileResultPage, Background),
    (GetCodeActions, Background),
    (GetCodeActionsResponse, Background),
    (GetCompletions, Background),
    (GetCompletionsResponse, Background),
    (GetDeclaration, Background),
    (GetDeclarationResponse, Background),
    (GetDefinition, Background),
    (GetDefinitionResponse, Background),
    (GetEditPredictionDefinition, Background),
    (GetEditPredictionDefinitionResponse, Background),
    (GetEditPredictionTypeDefinition, Background),
    (GetEditPredictionTypeDefinitionResponse, Background),
    (GetDocumentHighlights, Background),
    (GetDocumentHighlightsResponse, Background),
    (GetDocumentSymbols, Background),
    (GetDocumentSymbolsResponse, Background),
    (GetHover, Background),
    (GetHoverResponse, Background),
    (GetNotifications, Foreground),
    (GetNotificationsResponse, Foreground),
    (GetCrashFiles, Background),
    (GetCrashFilesResponse, Background),
    (GetPathMetadata, Background),
    (GetPathMetadataResponse, Background),
    (GetPermalinkToLine, Foreground),
    (GetProcesses, Background),
    (GetProcessesResponse, Background),
    (GetPermalinkToLineResponse, Foreground),
    (GetProjectSymbols, Background),
    (GetProjectSymbolsResponse, Background),
    (GetReferences, Background),
    (GetReferencesResponse, Background),
    (PrepareCallHierarchy, Background),
    (PrepareCallHierarchyResponse, Background),
    (GetIncomingCalls, Background),
    (GetIncomingCallsResponse, Background),
    (GetOutgoingCalls, Background),
    (GetOutgoingCallsResponse, Background),
    (GetSignatureHelp, Background),
    (GetSignatureHelpResponse, Background),
    (GetTypeDefinition, Background),
    (GetTypeDefinitionResponse, Background),
    (GetImplementation, Background),
    (GetImplementationResponse, Background),
    (OpenUnstagedDiff, Foreground),
    (OpenUnstagedDiffResponse, Foreground),
    (OpenUncommittedDiff, Foreground),
    (OpenUncommittedDiffResponse, Foreground),
    (GetUsers, Foreground),
    (GitGetBranches, Background),
    (GitBranchesResponse, Background),
    (Hello, Foreground),
    (HideToast, Background),
    (IncomingCall, Foreground),
    (InlayHints, Background),
    (InlayHintsResponse, Background),
    (SemanticTokens, Background),
    (SemanticTokensResponse, Background),
    (InstallExtension, Background),
    (InviteChannelMember, Foreground),
    (JoinChannel, Foreground),
    (JoinChannelBuffer, Foreground),
    (JoinChannelBufferResponse, Foreground),
    (JoinChannelChat, Foreground),
    (JoinChannelChatResponse, Foreground),
    (JoinProject, Foreground),
    (JoinProjectResponse, Foreground),
    (JoinRoom, Foreground),
    (JoinRoomResponse, Foreground),
    (LanguageServerLog, Foreground),
    (LanguageServerPromptRequest, Foreground),
    (LanguageServerPromptResponse, Foreground),
    (LeaveChannelBuffer, Background),
    (LeaveChannelChat, Foreground),
    (LeaveProject, Foreground),
    (LeaveRoom, Foreground),
    (LinkedEditingRange, Background),
    (LinkedEditingRangeResponse, Background),
    (ListRemoteDirectory, Background),
    (ListRemoteDirectoryResponse, Background),
    (ListToolchains, Foreground),
    (ListToolchainsResponse, Foreground),
    (LoadCommitDiff, Foreground),
    (LoadCommitDiffResponse, Foreground),
    (LspExtExpandMacro, Background),
    (LspExtExpandMacroResponse, Background),
    (LspExtOpenDocs, Background),
    (LspExtOpenDocsResponse, Background),
    (LspExtRunnables, Background),
    (LspExtRunnablesResponse, Background),
    (LspExtSwitchSourceHeader, Background),
    (LspExtSwitchSourceHeaderResponse, Background),
    (LspExtGoToParentModule, Background),
    (LspExtGoToParentModuleResponse, Background),
    (LspExtCancelFlycheck, Background),
    (LspExtRunFlycheck, Background),
    (LspExtClearFlycheck, Background),
    (MarkNotificationRead, Foreground),
    (MoveChannel, Foreground),
    (ReorderChannel, Foreground),
    (LspQuery, Background),
    (LspQueryResponse, Background),
    (OnTypeFormatting, Background),
    (OnTypeFormattingResponse, Background),
    (OpenBufferById, Background),
    (OpenBufferByPath, Background),
    (OpenBoundedBufferByPath, Background),
    (OpenBoundedBufferResponse, Background),
    (OpenImageByPath, Background),
    (OpenBufferForSymbol, Background),
    (OpenBufferForSymbolResponse, Background),
    (OpenBufferResponse, Background),
    (OpenImageResponse, Background),
    (OpenCommitMessageBuffer, Background),
    (OpenNewBuffer, Foreground),
    (OpenServerSettings, Foreground),
    (PerformRename, Background),
    (PerformRenameResponse, Background),
    (Ping, Foreground),
    (PrepareRename, Background),
    (PrepareRenameResponse, Background),
    (ProjectEntryResponse, Foreground),
    (RefreshInlayHints, Background),
    (RefreshSemanticTokens, Background),
    (RefreshDocumentColors, Background),
    (RefreshDocumentLinks, Background),
    (RefreshFoldingRanges, Background),
    (RefreshDocumentSymbols, Background),
    (RegisterBufferWithLanguageServers, Background),
    (RejoinChannelBuffers, Foreground),
    (RejoinChannelBuffersResponse, Foreground),
    (RejoinRemoteProjects, Foreground),
    (RejoinRemoteProjectsResponse, Foreground),
    (RejoinRoom, Foreground),
    (RejoinRoomResponse, Foreground),
    (ReloadBuffers, Foreground),
    (ReloadBuffersResponse, Foreground),
    (RemoveChannelMember, Foreground),
    (RemoveChannelMessage, Foreground),
    (RemoveContact, Foreground),
    (RemoveProjectCollaborator, Foreground),
    (RemoveWorktree, Foreground),
    (RenameChannel, Foreground),
    (RenameChannelResponse, Foreground),
    (RenameProjectEntry, Foreground),
    (RequestContact, Foreground),
    (ResolveCompletionDocumentation, Background),
    (ResolveCompletionDocumentationResponse, Background),
    (ResolveInlayHint, Background),
    (ResolveInlayHintResponse, Background),
    (GetDocumentColor, Background),
    (GetDocumentColorResponse, Background),
    (GetColorPresentation, Background),
    (GetColorPresentationResponse, Background),
    (GetDocumentLinks, Background),
    (GetDocumentLinksResponse, Background),
    (ResolveDocumentLink, Background),
    (ResolveDocumentLinkResponse, Background),
    (GetFoldingRanges, Background),
    (GetFoldingRangesResponse, Background),
    (RefreshCodeLens, Background),
    (GetCodeLens, Background),
    (GetCodeLensResponse, Background),
    (ResolveCodeAction, Background),
    (ResolveCodeActionResponse, Background),
    (RespondToChannelInvite, Foreground),
    (RespondToContactRequest, Foreground),
    (RestartLanguageServers, Foreground),
    (StopLanguageServers, Background),
    (RoomUpdated, Foreground),
    (SaveBuffer, Foreground),
    (SendChannelMessage, Background),
    (SendChannelMessageResponse, Background),
    (SetChannelMemberRole, Foreground),
    (SetChannelVisibility, Foreground),
    (SetRoomParticipantRole, Foreground),
    (ShareProject, Foreground),
    (ShareProjectResponse, Foreground),
    (ShowContacts, Foreground),
    (ShutdownRemoteServer, Foreground),
    (Stage, Background),
    (StartLanguageServer, Foreground),
    (SubscribeToChannels, Foreground),
    (SyncExtensions, Background),
    (SyncExtensionsResponse, Background),
    (BreakpointsForFile, Background),
    (ToggleBreakpoint, Foreground),
    (SynchronizeBuffers, Foreground),
    (SynchronizeBuffersResponse, Foreground),
    (TaskContext, Background),
    (TaskContextForLocation, Background),
    (TelemetryEvent, Background),
    (Test, Foreground),
    (Toast, Background),
    (Unfollow, Foreground),
    (UnshareProject, Foreground),
    (Unstage, Background),
    (Stash, Background),
    (StashPop, Background),
    (StashApply, Background),
    (StashDrop, Background),
    (UpdateBuffer, Foreground),
    (UpdateBufferFile, Foreground),
    (UpdateChannelBuffer, Foreground),
    (UpdateChannelBufferCollaborators, Foreground),
    (UpdateChannelMessage, Foreground),
    (UpdateChannels, Foreground),
    (UpdateContacts, Foreground),
    (UpdateDiagnosticSummary, Foreground),
    (UpdateDiffBases, Foreground),
    (UpdateFollowers, Foreground),
    (UpdateGitBranch, Background),
    (UpdateLanguageServer, Foreground),
    (UpdateNotification, Foreground),
    (UpdateParticipantLocation, Foreground),
    (UpdateProject, Foreground),
    (UpdateProjectCollaborator, Foreground),
    (UpdateUserChannels, Foreground),
    (UpdateWorktree, Foreground),
    (UpdateWorktreeSettings, Foreground),
    (UpdateUserSettings, Background),
    (UpdateRepository, Foreground),
    (RemoveRepository, Foreground),
    (UsersResponse, Foreground),
    (GitReset, Background),
    (GitDeleteBranch, Background),
    (GitCheckoutFiles, Background),
    (GitAddPathToGitignore, Background),
    (GitAddPathToGitInfoExclude, Background),
    (GitShow, Background),
    (GitCommitDetails, Background),
    (GitCreateCheckpoint, Background),
    (GitCreateCheckpointResponse, Background),
    (GitCreateArchiveCheckpoint, Background),
    (GitCreateArchiveCheckpointResponse, Background),
    (GitRestoreCheckpoint, Background),
    (GitRestoreArchiveCheckpoint, Background),
    (GitCompareCheckpoints, Background),
    (GitCompareCheckpointsResponse, Background),
    (GitDiffCheckpoints, Background),
    (GitDiffCheckpointsResponse, Background),
    (SetIndexText, Background),
    (Push, Background),
    (Fetch, Background),
    (GetRemotes, Background),
    (GetRemotesResponse, Background),
    (Pull, Background),
    (RemoteMessageResponse, Background),
    (AskPassRequest, Background),
    (AskPassResponse, Background),
    (GitCreateRemote, Background),
    (GitRemoveRemote, Background),
    (GitCreateBranch, Background),
    (GitChangeBranch, Background),
    (GitRenameBranch, Background),
    (TrustWorktrees, Background),
    (RestrictWorktrees, Background),
    (CheckForPushedCommits, Background),
    (CheckForPushedCommitsResponse, Background),
    (GitDiff, Background),
    (GitDiffResponse, Background),
    (GitInit, Background),
    (GetDebugAdapterBinary, Background),
    (DebugAdapterBinary, Background),
    (RunDebugLocators, Background),
    (DebugRequest, Background),
    (LogToDebugConsole, Background),
    (GetDocumentDiagnostics, Background),
    (GetDocumentDiagnosticsResponse, Background),
    (PullWorkspaceDiagnostics, Background),
    (GetDefaultBranch, Background),
    (GetDefaultBranchResponse, Background),
    (GetTreeDiff, Background),
    (GetTreeDiffResponse, Background),
    (GetBlobContent, Background),
    (GetBlobContentResponse, Background),
    (LoadCommitTemplate, Background),
    (LoadCommitTemplateResponse, Background),
    (GitClone, Background),
    (GitCloneResponse, Background),
    (ToggleLspLogs, Background),
    (GetDirectoryEnvironment, Background),
    (DirectoryEnvironment, Background),
    (GetAgentServerCommand, Background),
    (AgentServerCommand, Background),
    (GetContextServerCommand, Background),
    (ContextServerCommand, Background),
    (ExternalAgentsUpdated, Background),
    (ExternalExtensionAgentsUpdated, Background),
    (ExternalAgentLoadingStatusUpdated, Background),
    (NewExternalAgentVersionAvailable, Background),
    (GetCollaborationFilePolicy, Foreground),
    (GetCollaborationFilePolicyResponse, Foreground),
    (CollaborationFilePolicyInvalidated, Foreground),
    (DoctorRemoteServer, Foreground),
    (DoctorRemoteServerResponse, Foreground),
    (RemoteStarted, Background),
    (GitGetWorktrees, Background),
    (GitGetHeadSha, Background),
    (GitGetHeadShaResponse, Background),
    (GitEditRef, Background),
    (GitRepairWorktrees, Background),
    (GetCommitData, Background),
    (GetCommitDataResponse, Background),
    (GetInitialGraphData, Background),
    (GetInitialGraphDataResponse, Background),
    (SearchCommits, Background),
    (SearchCommitsResponse, Background),
    (GitWorktreesResponse, Background),
    (GitCreateWorktree, Background),
    (GitRemoveWorktree, Background),
    (GitRenameWorktree, Background),
    (GitWorktreeCreatedAt, Background),
    (GitWorktreeCreatedAtResponse, Background),
    (FindSearchCandidatesChunk, Background),
    (FindSearchCandidatesCancelled, Background),
    (SpawnKernel, Background),
    (SpawnKernelResponse, Background),
    (KillKernel, Background),
    (GetRemoteProfilingData, Background),
    (GetRemoteProfilingDataResponse, Background),
);

request_messages!(
    (AllocateWorktreeId, AllocateWorktreeIdResponse),
    (ApplyCodeAction, ApplyCodeActionResponse),
    (
        ApplyCompletionAdditionalEdits,
        ApplyCompletionAdditionalEditsResponse
    ),
    (Call, Ack),
    (CancelCargoWorkspace, Ack),
    (CancelCall, Ack),
    (Commit, Ack),
    (RunGitHook, Ack),
    (CopyProjectEntry, ProjectEntryResponse),
    (CreateChannel, CreateChannelResponse),
    (CreateProjectEntry, ProjectEntryResponse),
    (CreateRoom, CreateRoomResponse),
    (DeclineCall, Ack),
    (DeleteChannel, Ack),
    (DeleteProjectEntry, ProjectEntryResponse),
    (TrashProjectEntry, TrashProjectEntryResponse),
    (RestoreProjectEntry, RestoreProjectEntryResponse),
    (DownloadFileByPath, DownloadFileResponse),
    (ExpandProjectEntry, ExpandProjectEntryResponse),
    (ExpandAllForProjectEntry, ExpandAllForProjectEntryResponse),
    (Follow, FollowResponse),
    (ApplyCodeActionKind, ApplyCodeActionKindResponse),
    (FormatBuffers, FormatBuffersResponse),
    (FuzzySearchUsers, UsersResponse),
    (GetChannelMembers, GetChannelMembersResponse),
    (GetChannelMessages, GetChannelMessagesResponse),
    (GetChannelMessagesById, GetChannelMessagesResponse),
    (GetCodeActions, GetCodeActionsResponse),
    (GetCargoWorkspace, CargoWorkspaceResponse),
    (GetRustTestDiscovery, RustTestDiscoveryResponse),
    (CancelRustTestDiscovery, Ack),
    (GetJvmTestDiscovery, JvmTestDiscoveryResponse),
    (GetJdtlsProjectSnapshot, JdtlsProjectSnapshotResponse),
    (GetKotlinWorkspaceSnapshot, KotlinWorkspaceSnapshotResponse),
    (SetJvmProjectConsumers, JvmProjectConsumersResponse),
    (CancelJvmTestDiscovery, Ack),
    (ResolveJvmTestAction, JvmTestActionPlanResponse),
    (UpdateJvmTestRun, Ack),
    (ResolveRustTestAction, RustTestActionPlanResponse),
    (UpdateRustTestRun, Ack),
    (
        GetStructuredExecutionSnapshot,
        StructuredExecutionSnapshotPage
    ),
    (GetStructuredExecutionEvents, StructuredExecutionEventChunk),
    (GetSourceCoverage, SourceCoverageResponse),
    (InterpretRustCoverageArtifact, SourceCoverageResponse),
    (ImportProfileResults, ProfileResultPage),
    (PrepareProfileTaskBinding, ProfileTaskBindingResponse),
    (GetProfileTaskBinding, ProfileTaskBindingResponse),
    (CancelProfileTaskBinding, Ack),
    (GetProfileResultPage, ProfileResultPage),
    (CancelProfileImport, Ack),
    (GetCompletions, GetCompletionsResponse),
    (GetDefinition, GetDefinitionResponse),
    (
        GetEditPredictionDefinition,
        GetEditPredictionDefinitionResponse
    ),
    (
        GetEditPredictionTypeDefinition,
        GetEditPredictionTypeDefinitionResponse
    ),
    (GetDeclaration, GetDeclarationResponse),
    (GetImplementation, GetImplementationResponse),
    (GetDocumentHighlights, GetDocumentHighlightsResponse),
    (GetDocumentSymbols, GetDocumentSymbolsResponse),
    (GetHover, GetHoverResponse),
    (GetNotifications, GetNotificationsResponse),
    (GetProjectSymbols, GetProjectSymbolsResponse),
    (GetReferences, GetReferencesResponse),
    (PrepareCallHierarchy, PrepareCallHierarchyResponse),
    (GetIncomingCalls, GetIncomingCallsResponse),
    (GetOutgoingCalls, GetOutgoingCallsResponse),
    (GetSignatureHelp, GetSignatureHelpResponse),
    (OpenUnstagedDiff, OpenUnstagedDiffResponse),
    (OpenUncommittedDiff, OpenUncommittedDiffResponse),
    (GetTypeDefinition, GetTypeDefinitionResponse),
    (LinkedEditingRange, LinkedEditingRangeResponse),
    (ListRemoteDirectory, ListRemoteDirectoryResponse),
    (GetUsers, UsersResponse),
    (IncomingCall, Ack),
    (InlayHints, InlayHintsResponse),
    (SemanticTokens, SemanticTokensResponse),
    (GetCodeLens, GetCodeLensResponse),
    (InviteChannelMember, Ack),
    (JoinChannel, JoinRoomResponse),
    (JoinChannelBuffer, JoinChannelBufferResponse),
    (JoinChannelChat, JoinChannelChatResponse),
    (OpenCollaborativeChannel, OpenCollaborativeChannelResponse),
    (CloseCollaborativeChannel, Ack),
    (
        GetCollaborativeMessageWindow,
        GetCollaborativeMessageWindowResponse
    ),
    (
        ApplyCollaborativeMessageOperation,
        ApplyCollaborativeMessageOperationResponse
    ),
    (JoinProject, JoinProjectResponse),
    (JoinRoom, JoinRoomResponse),
    (LeaveChannelBuffer, Ack),
    (LeaveRoom, Ack),
    (LoadCommitDiff, LoadCommitDiffResponse),
    (MarkNotificationRead, Ack),
    (MoveChannel, Ack),
    (OnTypeFormatting, OnTypeFormattingResponse),
    (OpenBufferById, OpenBufferResponse),
    (OpenBufferByPath, OpenBufferResponse),
    (OpenBoundedBufferByPath, OpenBoundedBufferResponse),
    (OpenImageByPath, OpenImageResponse),
    (OpenBufferForSymbol, OpenBufferForSymbolResponse),
    (OpenCommitMessageBuffer, OpenBufferResponse),
    (OpenNewBuffer, OpenBufferResponse),
    (PerformRename, PerformRenameResponse),
    (Ping, Ack),
    (PrepareRename, PrepareRenameResponse),
    (RefreshInlayHints, Ack),
    (RefreshSemanticTokens, Ack),
    (RefreshCodeLens, Ack),
    (RefreshDocumentColors, Ack),
    (RefreshDocumentLinks, Ack),
    (RefreshFoldingRanges, Ack),
    (RefreshDocumentSymbols, Ack),
    (RejoinChannelBuffers, RejoinChannelBuffersResponse),
    (RejoinRoom, RejoinRoomResponse),
    (ReloadBuffers, ReloadBuffersResponse),
    (RemoveChannelMember, Ack),
    (RemoveChannelMessage, Ack),
    (UpdateChannelMessage, Ack),
    (RemoveContact, Ack),
    (RenameChannel, RenameChannelResponse),
    (RenameProjectEntry, ProjectEntryResponse),
    (ReorderChannel, Ack),
    (RequestContact, Ack),
    (
        ResolveCompletionDocumentation,
        ResolveCompletionDocumentationResponse
    ),
    (ResolveInlayHint, ResolveInlayHintResponse),
    (ResolveCodeAction, ResolveCodeActionResponse),
    (GetDocumentColor, GetDocumentColorResponse),
    (GetDocumentLinks, GetDocumentLinksResponse),
    (ResolveDocumentLink, ResolveDocumentLinkResponse),
    (GetFoldingRanges, GetFoldingRangesResponse),
    (GetColorPresentation, GetColorPresentationResponse),
    (RespondToChannelInvite, Ack),
    (RespondToContactRequest, Ack),
    (SaveBuffer, BufferSaved),
    (Stage, Ack),
    (FindSearchCandidates, Ack),
    (SendChannelMessage, SendChannelMessageResponse),
    (SetChannelMemberRole, Ack),
    (SetChannelVisibility, Ack),
    (ShareProject, ShareProjectResponse),
    (SynchronizeBuffers, SynchronizeBuffersResponse),
    (TaskContextForLocation, TaskContext),
    (Test, Test),
    (Unstage, Ack),
    (Stash, Ack),
    (StashPop, Ack),
    (StashApply, Ack),
    (StashDrop, Ack),
    (UpdateBuffer, Ack),
    (UpdateParticipantLocation, Ack),
    (UpdateProject, Ack),
    (UpdateWorktree, Ack),
    (UpdateRepository, Ack),
    (RemoveRepository, Ack),
    (LspExtExpandMacro, LspExtExpandMacroResponse),
    (LspExtOpenDocs, LspExtOpenDocsResponse),
    (LspExtRunnables, LspExtRunnablesResponse),
    (SetRoomParticipantRole, Ack),
    (BlameBuffer, BlameBufferResponse),
    (RejoinRemoteProjects, RejoinRemoteProjectsResponse),
    (LspQuery, Ack),
    (LspQueryResponse, Ack),
    (RestartLanguageServers, Ack),
    (StopLanguageServers, Ack),
    (LspExtSwitchSourceHeader, LspExtSwitchSourceHeaderResponse),
    (LspExtGoToParentModule, LspExtGoToParentModuleResponse),
    (LspExtCancelFlycheck, Ack),
    (LspExtRunFlycheck, Ack),
    (LspExtClearFlycheck, Ack),
    (AddWorktree, AddWorktreeResponse),
    (ShutdownRemoteServer, Ack),
    (RemoveWorktree, Ack),
    (OpenServerSettings, OpenBufferResponse),
    (GetPermalinkToLine, GetPermalinkToLineResponse),
    (FlushBufferedMessages, Ack),
    (LanguageServerPromptRequest, LanguageServerPromptResponse),
    (GitGetBranches, GitBranchesResponse),
    (UpdateGitBranch, Ack),
    (ListToolchains, ListToolchainsResponse),
    (ActivateToolchain, Ack),
    (ActiveToolchain, ActiveToolchainResponse),
    (ResolveToolchain, ResolveToolchainResponse),
    (GetPathMetadata, GetPathMetadataResponse),
    (GetCrashFiles, GetCrashFilesResponse),
    (CancelLanguageServerWork, Ack),
    (SyncExtensions, SyncExtensionsResponse),
    (InstallExtension, Ack),
    (RegisterBufferWithLanguageServers, Ack),
    (GitShow, GitCommitDetails),
    (GitCreateCheckpoint, GitCreateCheckpointResponse),
    (
        GitCreateArchiveCheckpoint,
        GitCreateArchiveCheckpointResponse
    ),
    (GitRestoreCheckpoint, Ack),
    (GitRestoreArchiveCheckpoint, Ack),
    (GitCompareCheckpoints, GitCompareCheckpointsResponse),
    (GitDiffCheckpoints, GitDiffCheckpointsResponse),
    (GitReset, Ack),
    (GitDeleteBranch, Ack),
    (GitCheckoutFiles, Ack),
    (GitAddPathToGitignore, Ack),
    (GitAddPathToGitInfoExclude, Ack),
    (SetIndexText, Ack),
    (Push, RemoteMessageResponse),
    (Fetch, RemoteMessageResponse),
    (GetRemotes, GetRemotesResponse),
    (Pull, RemoteMessageResponse),
    (AskPassRequest, AskPassResponse),
    (GitCreateRemote, Ack),
    (GitRemoveRemote, Ack),
    (GitCreateBranch, Ack),
    (GitChangeBranch, Ack),
    (GitRenameBranch, Ack),
    (CheckForPushedCommits, CheckForPushedCommitsResponse),
    (GitDiff, GitDiffResponse),
    (GitInit, Ack),
    (ToggleBreakpoint, Ack),
    (GetDebugAdapterBinary, DebugAdapterBinary),
    (RunDebugLocators, DebugRequest),
    (GetDocumentDiagnostics, GetDocumentDiagnosticsResponse),
    (PullWorkspaceDiagnostics, Ack),
    (GetDefaultBranch, GetDefaultBranchResponse),
    (GetBlobContent, GetBlobContentResponse),
    (LoadCommitTemplate, LoadCommitTemplateResponse),
    (GetTreeDiff, GetTreeDiffResponse),
    (GitClone, GitCloneResponse),
    (ToggleLspLogs, Ack),
    (GetDirectoryEnvironment, DirectoryEnvironment),
    (GetProcesses, GetProcessesResponse),
    (GetAgentServerCommand, AgentServerCommand),
    (GetContextServerCommand, ContextServerCommand),
    (
        GetCollaborationFilePolicy,
        GetCollaborationFilePolicyResponse
    ),
    (DoctorRemoteServer, DoctorRemoteServerResponse),
    (RemoteStarted, Ack),
    (GitGetWorktrees, GitWorktreesResponse),
    (GitGetHeadSha, GitGetHeadShaResponse),
    (GitEditRef, Ack),
    (GitRepairWorktrees, Ack),
    (GetCommitData, GetCommitDataResponse),
    (GetInitialGraphData, GetInitialGraphDataResponse),
    (SearchCommits, SearchCommitsResponse),
    (GitCreateWorktree, Ack),
    (GitRemoveWorktree, Ack),
    (GitRenameWorktree, Ack),
    (GitWorktreeCreatedAt, GitWorktreeCreatedAtResponse),
    (TrustWorktrees, Ack),
    (RestrictWorktrees, Ack),
    (FindSearchCandidatesChunk, Ack),
    (SpawnKernel, SpawnKernelResponse),
    (KillKernel, Ack),
    (GetRemoteProfilingData, GetRemoteProfilingDataResponse),
);

lsp_messages!(
    (GetReferences, GetReferencesResponse, true),
    (PrepareCallHierarchy, PrepareCallHierarchyResponse, true),
    (GetIncomingCalls, GetIncomingCallsResponse, true),
    (GetOutgoingCalls, GetOutgoingCallsResponse, true),
    (GetDocumentColor, GetDocumentColorResponse, true),
    (GetFoldingRanges, GetFoldingRangesResponse, true),
    (GetDocumentSymbols, GetDocumentSymbolsResponse, true),
    (GetDocumentLinks, GetDocumentLinksResponse, true),
    (GetHover, GetHoverResponse, true),
    (GetCodeActions, GetCodeActionsResponse, true),
    (GetSignatureHelp, GetSignatureHelpResponse, true),
    (GetCodeLens, GetCodeLensResponse, true),
    (GetDocumentDiagnostics, GetDocumentDiagnosticsResponse, true),
    (GetDefinition, GetDefinitionResponse, true),
    (
        GetEditPredictionDefinition,
        GetEditPredictionDefinitionResponse,
        true
    ),
    (
        GetEditPredictionTypeDefinition,
        GetEditPredictionTypeDefinitionResponse,
        true
    ),
    (GetDeclaration, GetDeclarationResponse, true),
    (GetTypeDefinition, GetTypeDefinitionResponse, true),
    (GetImplementation, GetImplementationResponse, true),
    (InlayHints, InlayHintsResponse, false),
    (SemanticTokens, SemanticTokensResponse, true)
);

entity_messages!(
    {project_id, ShareProject},
    AddProjectCollaborator,
    AddWorktree,
    AllocateWorktreeId,
    GetCollaborationFilePolicy,
    CollaborationFilePolicyInvalidated,
    ApplyCodeAction,
    ApplyCompletionAdditionalEdits,
    BlameBuffer,
    BufferReloaded,
    BufferSaved,
    CancelCargoWorkspace,
    CloseBuffer,
    Commit,
    RunGitHook,
    GetColorPresentation,
    CopyProjectEntry,
    CreateBufferForPeer,
    CreateFileForPeer,
    CreateImageForPeer,
    CreateProjectEntry,
    GetDocumentColor,
    GetDocumentLinks,
    ResolveDocumentLink,
    GetFoldingRanges,
    DeleteProjectEntry,
    TrashProjectEntry,
    RestoreProjectEntry,
    ExpandProjectEntry,
    ExpandAllForProjectEntry,
    FindSearchCandidates,
    ApplyCodeActionKind,
    FormatBuffers,
    GetCodeActions,
    GetCargoWorkspace,
    GetRustTestDiscovery,
    CancelRustTestDiscovery,
    GetJvmTestDiscovery,
    GetJdtlsProjectSnapshot,
    GetKotlinWorkspaceSnapshot,
    SetJvmProjectConsumers,
    CancelJvmTestDiscovery,
    ResolveJvmTestAction,
    UpdateJvmTestRun,
    ResolveRustTestAction,
    UpdateRustTestRun,
    GetStructuredExecutionSnapshot,
    GetStructuredExecutionEvents,
    GetSourceCoverage,
    InterpretRustCoverageArtifact,
    ImportProfileResults,
    PrepareProfileTaskBinding,
    GetProfileTaskBinding,
    CancelProfileTaskBinding,
    GetProfileResultPage,
    CancelProfileImport,
    GetCodeLens,
    GetCompletions,
    GetDefinition,
    GetEditPredictionDefinition,
    GetEditPredictionTypeDefinition,
    GetDeclaration,
    GetImplementation,
    GetDocumentHighlights,
    GetDocumentSymbols,
    GetHover,
    GetProjectSymbols,
    GetReferences,
    PrepareCallHierarchy,
    GetIncomingCalls,
    GetOutgoingCalls,
    GetSignatureHelp,
    OpenUnstagedDiff,
    OpenUncommittedDiff,
    GetTypeDefinition,
    InlayHints,
    SemanticTokens,
    JoinProject,
    LeaveProject,
    SpawnKernel,
    KillKernel,
    LinkedEditingRange,
    LoadCommitDiff,
    LspQuery,
    LspQueryResponse,
    RestartLanguageServers,
    StopLanguageServers,
    OnTypeFormatting,
    OpenNewBuffer,
    OpenBufferById,
    OpenBufferByPath,
    OpenBoundedBufferByPath,
    OpenImageByPath,
    OpenBufferForSymbol,
    OpenCommitMessageBuffer,
    PerformRename,
    PrepareRename,
    RefreshInlayHints,
    RefreshSemanticTokens,
    RefreshCodeLens,
    RefreshDocumentColors,
    RefreshDocumentLinks,
    RefreshFoldingRanges,
    RefreshDocumentSymbols,
    ReloadBuffers,
    RemoveProjectCollaborator,
    RenameProjectEntry,
    ResolveCompletionDocumentation,
    ResolveInlayHint,
    ResolveCodeAction,
    SaveBuffer,
    Stage,
    StartLanguageServer,
    SynchronizeBuffers,
    TaskContextForLocation,
    UnshareProject,
    Unstage,
    Stash,
    StashPop,
    StashApply,
    StashDrop,
    UpdateBuffer,
    UpdateBufferFile,
    UpdateDiagnosticSummary,
    UpdateDiffBases,
    UpdateLanguageServer,
    UpdateProject,
    UpdateProjectCollaborator,
    UpdateWorktree,
    UpdateRepository,
    RemoveRepository,
    UpdateWorktreeSettings,
    UpdateUserSettings,
    LspExtExpandMacro,
    LspExtOpenDocs,
    LspExtRunnables,
    LspExtSwitchSourceHeader,
    LspExtGoToParentModule,
    LspExtCancelFlycheck,
    LspExtRunFlycheck,
    LspExtClearFlycheck,
    LanguageServerLog,
    TelemetryEvent,
    Toast,
    HideToast,
    OpenServerSettings,
    GetPermalinkToLine,
    LanguageServerPromptRequest,
    GitGetBranches,
    UpdateGitBranch,
    ListToolchains,
    ActivateToolchain,
    ActiveToolchain,
    ResolveToolchain,
    GetPathMetadata,
    GetProcesses,
    CancelLanguageServerWork,
    RegisterBufferWithLanguageServers,
    GitShow,
    GitCreateCheckpoint,
    GitRestoreCheckpoint,
    GitCompareCheckpoints,
    GitDiffCheckpoints,
    GitReset,
    GitDeleteBranch,
    GitCheckoutFiles,
    GitAddPathToGitignore,
    GitAddPathToGitInfoExclude,
    SetIndexText,
    ToggleLspLogs,
    GetDirectoryEnvironment,

    Push,
    Fetch,
    GetRemotes,
    Pull,
    AskPassRequest,
    GitChangeBranch,
    GitRenameBranch,
    GitCreateBranch,
    GitCreateRemote,
    GitRemoveRemote,
    CheckForPushedCommits,
    GitDiff,
    GitInit,
    BreakpointsForFile,
    ToggleBreakpoint,
    RunDebugLocators,
    GetDebugAdapterBinary,
    LogToDebugConsole,
    GetDocumentDiagnostics,
    PullWorkspaceDiagnostics,
    GetDefaultBranch,
    GetTreeDiff,
    GetBlobContent,
    LoadCommitTemplate,
    GitClone,
    GetAgentServerCommand,
    GetContextServerCommand,
    ExternalAgentsUpdated,
    ExternalExtensionAgentsUpdated,
    ExternalAgentLoadingStatusUpdated,
    NewExternalAgentVersionAvailable,
    GitGetWorktrees,
    GitGetHeadSha,
    GitEditRef,
    GitRepairWorktrees,
    GetCommitData,
    GetInitialGraphData,
    SearchCommits,
    GitCreateArchiveCheckpoint,
    GitRestoreArchiveCheckpoint,
    GitCreateWorktree,
    GitRemoveWorktree,
    GitRenameWorktree,
    GitWorktreeCreatedAt,
    TrustWorktrees,
    RestrictWorktrees,
    FindSearchCandidatesChunk,
    FindSearchCandidatesCancelled,
    DownloadFileByPath,
    GetRemoteProfilingData
);

entity_messages!(
    {channel_id, Channel},
    ChannelMessageSent,
    ChannelMessageUpdate,
    RemoveChannelMessage,
    UpdateChannelMessage,
    UpdateChannelBuffer,
    UpdateChannelBufferCollaborators,
);

impl From<Timestamp> for SystemTime {
    fn from(val: Timestamp) -> Self {
        UNIX_EPOCH
            .checked_add(Duration::new(val.seconds, val.nanos))
            .unwrap()
    }
}

impl From<SystemTime> for Timestamp {
    fn from(time: SystemTime) -> Self {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
        Self {
            seconds: duration.as_secs(),
            nanos: duration.subsec_nanos(),
        }
    }
}

impl From<u128> for Nonce {
    fn from(nonce: u128) -> Self {
        let upper_half = (nonce >> 64) as u64;
        let lower_half = nonce as u64;
        Self {
            upper_half,
            lower_half,
        }
    }
}

impl From<Nonce> for u128 {
    fn from(nonce: Nonce) -> Self {
        let upper_half = (nonce.upper_half as u128) << 64;
        let lower_half = nonce.lower_half as u128;
        upper_half | lower_half
    }
}

#[cfg(any(test, feature = "test-support"))]
pub const MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE: usize = 2;
#[cfg(not(any(test, feature = "test-support")))]
pub const MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE: usize = 256;

pub fn split_worktree_update(mut message: UpdateWorktree) -> impl Iterator<Item = UpdateWorktree> {
    let mut done = false;

    iter::from_fn(move || {
        if done {
            return None;
        }

        let updated_entries_chunk_size = cmp::min(
            message.updated_entries.len(),
            MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE,
        );
        let updated_entries: Vec<_> = message
            .updated_entries
            .drain(..updated_entries_chunk_size)
            .collect();

        let removed_entries_chunk_size = cmp::min(
            message.removed_entries.len(),
            MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE,
        );
        let removed_entries = message
            .removed_entries
            .drain(..removed_entries_chunk_size)
            .collect();

        let mut updated_repositories = Vec::new();
        let mut limit = MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE;
        while let Some(repo) = message.updated_repositories.first_mut() {
            let updated_statuses_limit = cmp::min(repo.updated_statuses.len(), limit);
            let removed_statuses_limit = cmp::min(repo.removed_statuses.len(), limit);

            updated_repositories.push(RepositoryEntry {
                repository_id: repo.repository_id,
                branch_summary: repo.branch_summary.clone(),
                updated_statuses: repo
                    .updated_statuses
                    .drain(..updated_statuses_limit)
                    .collect(),
                removed_statuses: repo
                    .removed_statuses
                    .drain(..removed_statuses_limit)
                    .collect(),
                current_merge_conflicts: repo.current_merge_conflicts.clone(),
            });
            if repo.removed_statuses.is_empty() && repo.updated_statuses.is_empty() {
                message.updated_repositories.remove(0);
            }
            limit = limit.saturating_sub(removed_statuses_limit + updated_statuses_limit);
            if limit == 0 {
                break;
            }
        }

        done = message.updated_entries.is_empty()
            && message.removed_entries.is_empty()
            && message.updated_repositories.is_empty();

        let removed_repositories = if done {
            mem::take(&mut message.removed_repositories)
        } else {
            Default::default()
        };

        Some(UpdateWorktree {
            project_id: message.project_id,
            worktree_id: message.worktree_id,
            root_name: message.root_name.clone(),
            abs_path: message.abs_path.clone(),
            root_repo_common_dir: message.root_repo_common_dir.clone(),
            root_repo_is_linked_worktree: message.root_repo_is_linked_worktree,
            updated_entries,
            removed_entries,
            scan_id: message.scan_id,
            is_last_update: done && message.is_last_update,
            updated_repositories,
            removed_repositories,
        })
    })
}

pub fn split_repository_update(
    mut update: UpdateRepository,
) -> impl Iterator<Item = UpdateRepository> {
    let mut updated_statuses_iter = mem::take(&mut update.updated_statuses).into_iter().fuse();
    let mut removed_statuses_iter = mem::take(&mut update.removed_statuses).into_iter().fuse();
    let branch_list = mem::take(&mut update.branch_list);
    let branch_list_error = update.branch_list_error.take();
    std::iter::from_fn({
        let update = update.clone();
        move || {
            let updated_statuses = updated_statuses_iter
                .by_ref()
                .take(MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE)
                .collect::<Vec<_>>();
            let removed_statuses = removed_statuses_iter
                .by_ref()
                .take(MAX_WORKTREE_UPDATE_MAX_CHUNK_SIZE)
                .collect::<Vec<_>>();
            if updated_statuses.is_empty() && removed_statuses.is_empty() {
                return None;
            }
            Some(UpdateRepository {
                updated_statuses,
                removed_statuses,
                branch_list: Vec::new(),
                branch_list_error: None,
                is_last_update: false,
                ..update.clone()
            })
        }
    })
    .chain([UpdateRepository {
        updated_statuses: Vec::new(),
        removed_statuses: Vec::new(),
        branch_list,
        branch_list_error,
        is_last_update: true,
        ..update
    }])
}

impl LspQuery {
    pub fn query_name_and_write_permissions(&self) -> (&str, bool) {
        match self.request {
            Some(lsp_query::Request::GetHover(_)) => ("GetHover", false),
            Some(lsp_query::Request::GetCodeActions(_)) => ("GetCodeActions", true),
            Some(lsp_query::Request::GetSignatureHelp(_)) => ("GetSignatureHelp", false),
            Some(lsp_query::Request::GetCodeLens(_)) => ("GetCodeLens", true),
            Some(lsp_query::Request::GetDocumentDiagnostics(_)) => {
                ("GetDocumentDiagnostics", false)
            }
            Some(lsp_query::Request::GetDefinition(_)) => ("GetDefinition", false),
            Some(lsp_query::Request::GetEditPredictionDefinition(_)) => {
                ("GetEditPredictionDefinition", false)
            }
            Some(lsp_query::Request::GetEditPredictionTypeDefinition(_)) => {
                ("GetEditPredictionTypeDefinition", false)
            }
            Some(lsp_query::Request::GetDeclaration(_)) => ("GetDeclaration", false),
            Some(lsp_query::Request::GetTypeDefinition(_)) => ("GetTypeDefinition", false),
            Some(lsp_query::Request::GetImplementation(_)) => ("GetImplementation", false),
            Some(lsp_query::Request::GetReferences(_)) => ("GetReferences", false),
            Some(lsp_query::Request::PrepareCallHierarchy(_)) => ("PrepareCallHierarchy", false),
            Some(lsp_query::Request::GetIncomingCalls(_)) => ("GetIncomingCalls", false),
            Some(lsp_query::Request::GetOutgoingCalls(_)) => ("GetOutgoingCalls", false),
            Some(lsp_query::Request::GetDocumentColor(_)) => ("GetDocumentColor", false),
            Some(lsp_query::Request::GetFoldingRanges(_)) => ("GetFoldingRanges", false),
            Some(lsp_query::Request::GetDocumentSymbols(_)) => ("GetDocumentSymbols", false),
            Some(lsp_query::Request::GetDocumentLinks(_)) => ("GetDocumentLinks", false),
            Some(lsp_query::Request::InlayHints(_)) => ("InlayHints", false),
            Some(lsp_query::Request::SemanticTokens(_)) => ("SemanticTokens", false),
            None => ("<unknown>", true),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn collaboration_policy_request() -> GetCollaborationFilePolicy {
        GetCollaborationFilePolicy {
            protocol_version: COLLABORATION_FILE_POLICY_VERSION,
            project_id: REMOTE_SERVER_PROJECT_ID,
            scopes: vec![CollaborationFilePolicyScope {
                worktree_id: 3,
                abs_path: "/workspace/.env".into(),
                root_name: ".env".into(),
                root_kind: CollaborationFilePolicyRootKind::File as i32,
                path_style: Some(CollaborationFilePolicyPathStyle::Posix as i32),
                path: String::new(),
            }],
        }
    }

    fn collaboration_policy_response(
        request: &GetCollaborationFilePolicy,
    ) -> GetCollaborationFilePolicyResponse {
        GetCollaborationFilePolicyResponse {
            protocol_version: COLLABORATION_FILE_POLICY_VERSION,
            project_id: request.project_id,
            checked_settings_revision: Some(0),
            results: request
                .scopes
                .iter()
                .map(|scope| CollaborationFilePolicyResult {
                    scope: Some(scope.clone()),
                    decision: CollaborationFilePolicyDecision::Allow as i32,
                })
                .collect(),
        }
    }

    fn collaboration_policy_wide_request() -> GetCollaborationFilePolicy {
        GetCollaborationFilePolicy {
            protocol_version: COLLABORATION_FILE_POLICY_VERSION,
            project_id: REMOTE_SERVER_PROJECT_ID,
            scopes: (1..=16)
                .map(|worktree_id| CollaborationFilePolicyScope {
                    worktree_id,
                    abs_path: "/".into(),
                    root_name: String::new(),
                    root_kind: CollaborationFilePolicyRootKind::Directory as i32,
                    path_style: Some(CollaborationFilePolicyPathStyle::Posix as i32),
                    path: "x".repeat(MAX_COLLABORATION_FILE_POLICY_PATH_BYTES - 1),
                })
                .collect(),
        }
    }

    #[test]
    fn collaboration_policy_protocol_registers_native_zero_scope_and_round_trips() {
        fn assert_request<
            T: RequestMessage<Response = GetCollaborationFilePolicyResponse>
                + EntityMessage<Entity = ShareProject>,
        >() {
        }
        fn assert_invalidation<T: EntityMessage<Entity = ShareProject>>() {}
        assert_request::<GetCollaborationFilePolicy>();
        assert_invalidation::<CollaborationFilePolicyInvalidated>();
        let request = collaboration_policy_request();
        request.validate().expect("bounded native project0 request");
        assert_eq!(request.remote_entity_id(), 0);
        assert_eq!(request.scopes.first().expect("single-file scope").path, "");
        let sender = PeerId { owner_id: 4, id: 5 };
        let original_sender = PeerId { owner_id: 7, id: 8 };
        let envelope = request
            .clone()
            .into_envelope(11, None, Some(original_sender));
        let decoded =
            Envelope::decode(envelope.encode_to_vec().as_slice()).expect("policy request envelope");
        let typed = build_typed_envelope(sender, std::time::Instant::now(), decoded.clone())
            .expect("registered typed policy request");
        assert_eq!(typed.sender_id(), sender);
        assert_eq!(typed.original_sender_id(), Some(original_sender));
        assert_eq!(typed.message_id(), 11);
        assert!(!typed.is_background());
        assert_eq!(typed.payload_type_name(), "GetCollaborationFilePolicy");
        assert_eq!(
            GetCollaborationFilePolicy::from_envelope(decoded),
            Some(request.clone())
        );
        let response = collaboration_policy_response(&request);
        response
            .validate_for_request(&request)
            .expect("present revision0");
        let envelope = response.clone().into_envelope(12, Some(11), None);
        let decoded = Envelope::decode(envelope.encode_to_vec().as_slice())
            .expect("policy response envelope");
        assert_eq!(decoded.responding_to, Some(11));
        assert_eq!(
            GetCollaborationFilePolicyResponse::from_envelope(decoded),
            Some(response)
        );
    }

    #[test]
    fn collaboration_policy_envelopes_use_exact_tags_and_preserve_old_wire() {
        fn check<T: EnvelopedMessage + Default + PartialEq>(wire: &[u8]) {
            let message = T::default();
            assert_eq!(
                message.clone().into_envelope(0, None, None).encode_to_vec(),
                wire
            );
            let decoded = Envelope::decode(wire).expect("golden policy envelope");
            assert_eq!(T::from_envelope(decoded), Some(message));
            assert!(Envelope::decode(&wire[..wire.len() - 1]).is_err());
        }
        check::<GetCollaborationFilePolicy>(&[0xa2, 0x21, 0]);
        check::<GetCollaborationFilePolicyResponse>(&[0xaa, 0x21, 0]);
        check::<CollaborationFilePolicyInvalidated>(&[0xb2, 0x21, 0]);
        check::<DoctorRemoteServer>(&[0x92, 0x21, 0]);
        check::<DoctorRemoteServerResponse>(&[0x9a, 0x21, 0]);
        check::<Ping>(&[0x3a, 0]);
        check::<UpdateProject>(&[0xe2, 0x02, 0]);
    }

    #[test]
    fn collaboration_policy_rejects_defaults_and_unsupported_versions() {
        assert!(GetCollaborationFilePolicy::default().validate().is_err());
        assert!(
            CollaborationFilePolicyInvalidated::default()
                .validate()
                .is_err()
        );
        let mut request = collaboration_policy_request();
        for version in [0, 2, u32::MAX] {
            request.protocol_version = version;
            assert!(request.validate().is_err());
        }
        request.protocol_version = COLLABORATION_FILE_POLICY_VERSION;
        let mut response = collaboration_policy_response(&request);
        for version in [0, 2, u32::MAX] {
            response.protocol_version = version;
            assert!(response.validate_for_request(&request).is_err());
        }
        request.scopes.clear();
        assert!(request.validate().is_err());
    }

    #[test]
    fn collaboration_policy_preserves_revision_presence_and_known_denial() {
        let request = collaboration_policy_request();
        let mut response = collaboration_policy_response(&request);
        response.checked_settings_revision = None;
        assert!(response.validate_for_request(&request).is_err());
        response.checked_settings_revision = Some(0);
        response
            .validate_for_request(&request)
            .expect("explicit checked zero");
        let decoded =
            GetCollaborationFilePolicyResponse::decode(response.encode_to_vec().as_slice())
                .expect("explicit revision0 roundtrip");
        assert_eq!(decoded.checked_settings_revision, Some(0));
        let absent = GetCollaborationFilePolicyResponse::decode(&[8, 1][..])
            .expect("versioned but absent revision");
        assert_eq!(absent.checked_settings_revision, None);
        let present = GetCollaborationFilePolicyResponse::decode(&[8, 1, 24, 0][..])
            .expect("present zero revision wire");
        assert_eq!(present.checked_settings_revision, Some(0));
        for decision in [0, 99] {
            response.results.first_mut().expect("result").decision = decision;
            assert!(response.validate_for_request(&request).is_err());
        }
        response.results.first_mut().expect("result").decision =
            CollaborationFilePolicyDecision::Deny as i32;
        response
            .validate_for_request(&request)
            .expect("known denial is valid protocol, never Allow");
        assert_eq!(
            response.results.first().expect("result").decision,
            CollaborationFilePolicyDecision::Deny as i32
        );
        response.project_id = 1;
        assert!(response.validate_for_request(&request).is_err());
    }

    #[test]
    fn collaboration_policy_requires_exact_order_cardinality_and_logical_uniqueness() {
        let mut request = collaboration_policy_request();
        let mut second = request.scopes.first().expect("first scope").clone();
        second.worktree_id = 4;
        request.scopes.push(second);
        request.validate().expect("two distinct scopes");
        let response = collaboration_policy_response(&request);
        response.validate_for_request(&request).expect("exact pair");
        let mut missing = response.clone();
        missing.results.pop();
        assert!(missing.validate_for_request(&request).is_err());
        let mut extra = response.clone();
        extra
            .results
            .push(response.results.first().expect("first result").clone());
        assert!(extra.validate_for_request(&request).is_err());
        let mut reordered = response.clone();
        reordered.results.reverse();
        assert!(reordered.validate_for_request(&request).is_err());
        let mut duplicate = response.clone();
        *duplicate.results.last_mut().expect("second result") =
            response.results.first().expect("first result").clone();
        assert!(duplicate.validate_for_request(&request).is_err());
        let mut no_scope = response.clone();
        no_scope.results.first_mut().expect("first result").scope = None;
        assert!(no_scope.validate_for_request(&request).is_err());
        let mut wrong_root = response;
        wrong_root
            .results
            .first_mut()
            .expect("first result")
            .scope
            .as_mut()
            .expect("scope")
            .abs_path = "/other".into();
        assert!(wrong_root.validate_for_request(&request).is_err());
        let mut duplicated_request = request.clone();
        let mut same_logical_scope = request.scopes.first().expect("first scope").clone();
        same_logical_scope.abs_path = "/different-root-metadata".into();
        duplicated_request.scopes.push(same_logical_scope);
        assert!(duplicated_request.validate().is_err());
    }

    #[test]
    fn collaboration_policy_checks_structural_path_shape_without_native_authority() {
        let valid = collaboration_policy_request();
        let posix_zero =
            CollaborationFilePolicyScope::decode(&[40, 0][..]).expect("explicit POSIX zero style");
        assert_eq!(posix_zero.path_style, Some(0));
        let absent = CollaborationFilePolicyScope::decode(&[][..]).expect("absent style wire");
        assert_eq!(absent.path_style, None);
        assert_eq!(CollaborationFilePolicyPathStyle::Posix as i32, 0);
        assert_eq!(CollaborationFilePolicyPathStyle::Windows as i32, 1);
        valid.validate().expect("present POSIX zero remains valid");
        for (root_kind, path_style) in [(0, Some(0)), (99, Some(0)), (2, None), (2, Some(99))] {
            let mut request = valid.clone();
            let scope = request.scopes.first_mut().expect("scope");
            scope.root_kind = root_kind;
            scope.path_style = path_style;
            assert!(request.validate().is_err());
        }
        let mut windows = valid.clone();
        let scope = windows.scopes.first_mut().expect("scope");
        scope.abs_path = r"C:\work\file.txt".into();
        scope.root_name = "file.txt".into();
        scope.path_style = Some(CollaborationFilePolicyPathStyle::Windows as i32);
        windows.validate().expect("structural Windows descriptor");
        let response = collaboration_policy_response(&windows);
        response
            .validate_for_request(&windows)
            .expect("unchanged Windows echo");
        for field in [0, 1, 2] {
            let mut request = valid.clone();
            let scope = request.scopes.first_mut().expect("scope");
            match field {
                0 => scope.abs_path = "private-canary\0root".into(),
                1 => scope.root_name = "private-canary\0name".into(),
                _ => scope.path = "private-canary\0path".into(),
            }
            let error = request
                .validate()
                .expect_err("NUL is invalid structure")
                .to_string();
            assert!(!error.contains("private-canary"));
        }
    }

    #[test]
    fn collaboration_policy_bounds_scope_count_and_utf8_path_bytes() {
        let mut request = collaboration_policy_request();
        let template = request.scopes.first().expect("scope").clone();
        request.scopes = (0..MAX_COLLABORATION_FILE_POLICY_SCOPES)
            .map(|index| CollaborationFilePolicyScope {
                worktree_id: index as u64,
                ..template.clone()
            })
            .collect();
        request.validate().expect("exact scope count bound");
        request.scopes.push(CollaborationFilePolicyScope {
            worktree_id: u64::MAX,
            ..template
        });
        assert!(request.validate().is_err());
        for field in [0, 1, 2] {
            let mut request = collaboration_policy_request();
            let scope = request.scopes.first_mut().expect("scope");
            let path = match field {
                0 => &mut scope.abs_path,
                1 => &mut scope.root_name,
                _ => &mut scope.path,
            };
            *path = "é".repeat(MAX_COLLABORATION_FILE_POLICY_PATH_BYTES / 2);
            assert_eq!(path.len(), MAX_COLLABORATION_FILE_POLICY_PATH_BYTES);
            request.validate().expect("UTF8 byte bound");
            let scope = request.scopes.first_mut().expect("scope");
            let path = match field {
                0 => &mut scope.abs_path,
                1 => &mut scope.root_name,
                _ => &mut scope.path,
            };
            path.push('a');
            assert!(request.validate().is_err());
        }
    }

    #[test]
    fn collaboration_policy_bounds_aggregate_paths_separately_from_encoded_payload() {
        let mut request = collaboration_policy_wide_request();
        validate_collaboration_file_policy_scopes(&request.scopes)
            .expect("exact aggregate byte bound");
        assert!(request.encoded_len() > MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES);
        assert_eq!(
            request
                .validate()
                .expect_err("encoding overhead bounded")
                .to_string(),
            "collaboration policy encoded payload exceeds limit"
        );
        request.scopes.first_mut().expect("scope").path.push('a');
        assert_eq!(
            request
                .validate()
                .expect_err("aggregate bound +1")
                .to_string(),
            "collaboration policy aggregate path bytes exceed limit"
        );
    }

    #[test]
    fn collaboration_policy_query_and_response_enforce_exact_encoded_byte_boundary() {
        let mut request = collaboration_policy_wide_request();
        let excess = request.encoded_len() - MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES;
        let path = &mut request.scopes.last_mut().expect("last scope").path;
        let remaining = path
            .len()
            .checked_sub(excess)
            .expect("fixture has spare path bytes");
        path.truncate(remaining);
        assert_eq!(
            request.encoded_len(),
            MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES
        );
        request.validate().expect("query exact encoded bound");
        request
            .scopes
            .last_mut()
            .expect("last scope")
            .path
            .push('a');
        assert_eq!(
            request.encoded_len(),
            MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES + 1
        );
        assert!(request.validate().is_err());
        let mut request = collaboration_policy_wide_request();
        let response = collaboration_policy_response(&request);
        let excess = response.encoded_len() - MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES;
        let path = &mut request.scopes.last_mut().expect("last scope").path;
        let remaining = path
            .len()
            .checked_sub(excess)
            .expect("fixture has spare path bytes");
        path.truncate(remaining);
        let response = collaboration_policy_response(&request);
        assert_eq!(
            response.encoded_len(),
            MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES
        );
        response
            .validate_for_request(&request)
            .expect("response exact encoded bound");
        request
            .scopes
            .last_mut()
            .expect("last scope")
            .path
            .push('a');
        let response = collaboration_policy_response(&request);
        assert_eq!(
            response.encoded_len(),
            MAX_COLLABORATION_FILE_POLICY_ENCODED_BYTES + 1
        );
        request
            .validate()
            .expect("query remains below its own bound");
        assert!(response.validate_for_request(&request).is_err());
    }

    #[test]
    fn collaboration_policy_invalidation_is_bounded_and_never_a_grant() {
        let mut invalidation = CollaborationFilePolicyInvalidated {
            protocol_version: COLLABORATION_FILE_POLICY_VERSION,
            project_id: REMOTE_SERVER_PROJECT_ID,
            checked_settings_revision: None,
            worktree_ids: Vec::new(),
        };
        invalidation
            .validate()
            .expect("unknown revision invalidates whole project");
        assert_eq!(invalidation.remote_entity_id(), 0);
        invalidation.checked_settings_revision = Some(0);
        invalidation.worktree_ids = vec![3, 4];
        invalidation.validate().expect("revision0 invalidation");
        let envelope = invalidation.clone().into_envelope(7, None, None);
        let decoded =
            Envelope::decode(envelope.encode_to_vec().as_slice()).expect("invalidation envelope");
        assert_eq!(
            CollaborationFilePolicyInvalidated::from_envelope(decoded),
            Some(invalidation.clone())
        );
        invalidation
            .validate()
            .expect("same revision remains a valid invalidation");
        invalidation.worktree_ids = (0..MAX_COLLABORATION_FILE_POLICY_SCOPES)
            .map(|index| index as u64)
            .collect();
        invalidation
            .validate()
            .expect("exact invalidation root count");
        invalidation.worktree_ids.push(u64::MAX);
        assert!(invalidation.validate().is_err());
        invalidation.worktree_ids = vec![3, 3];
        assert!(invalidation.validate().is_err());
        for version in [0, 2, u32::MAX] {
            invalidation.protocol_version = version;
            assert!(invalidation.validate().is_err());
        }
    }

    #[test]
    fn jvm_project_consumer_protocol_routes_current_scope_and_withdrawal() {
        fn assert_request<
            T: RequestMessage<Response = JvmProjectConsumersResponse> + EntityMessage,
        >() {
        }
        assert_request::<SetJvmProjectConsumers>();
        let roots = vec![
            ProjectPath {
                worktree_id: 3,
                path: String::new(),
            },
            ProjectPath {
                worktree_id: 9,
                path: "included build/应用".into(),
            },
        ];
        let request = SetJvmProjectConsumers {
            project_id: 7,
            protocol_version: 1,
            generation: 19,
            roots: roots.clone(),
        };
        assert_eq!(request.remote_entity_id(), 7);
        let envelope = request
            .clone()
            .into_envelope(12, None, Some(PeerId { owner_id: 1, id: 2 }));
        let decoded = Envelope::decode(envelope.encode_to_vec().as_slice())
            .expect("JVM consumer scope envelope");
        assert_eq!(decoded.original_sender_id, envelope.original_sender_id);
        assert_eq!(
            SetJvmProjectConsumers::from_envelope(decoded),
            Some(request)
        );

        let response = JvmProjectConsumersResponse {
            protocol_version: 1,
            generation: 19,
            accepted_roots: roots,
        };
        let envelope = response.clone().into_envelope(13, Some(12), None);
        let decoded = Envelope::decode(envelope.encode_to_vec().as_slice())
            .expect("JVM consumer acknowledgment envelope");
        assert_eq!(decoded.responding_to, Some(12));
        assert_eq!(
            JvmProjectConsumersResponse::from_envelope(decoded),
            Some(response)
        );

        let withdrawal = SetJvmProjectConsumers {
            project_id: 7,
            protocol_version: 1,
            generation: 20,
            roots: Vec::new(),
        };
        assert_eq!(
            SetJvmProjectConsumers::decode(withdrawal.encode_to_vec().as_slice())
                .expect("JVM consumer withdrawal"),
            withdrawal
        );
    }

    #[test]
    fn jvm_project_consumer_envelopes_preserve_existing_wire_numbers() {
        fn check<T: EnvelopedMessage + Default + PartialEq>(wire: &[u8]) {
            let message = T::default();
            assert_eq!(
                message.clone().into_envelope(0, None, None).encode_to_vec(),
                wire
            );
            let decoded = Envelope::decode(wire).expect("golden envelope");
            assert_eq!(T::from_envelope(decoded), Some(message));
            assert!(Envelope::decode(&wire[..wire.len() - 1]).is_err());
        }
        check::<SetJvmProjectConsumers>(&[0x82, 0x21, 0]);
        check::<JvmProjectConsumersResponse>(&[0x8a, 0x21, 0]);
        check::<GetJdtlsProjectSnapshot>(&[0xe2, 0x20, 0]);
        check::<JdtlsProjectSnapshotResponse>(&[0xea, 0x20, 0]);
        check::<GetKotlinWorkspaceSnapshot>(&[0xf2, 0x20, 0]);
        check::<KotlinWorkspaceSnapshotResponse>(&[0xfa, 0x20, 0]);
        check::<GetCargoWorkspace>(&[0xf2, 0x1d, 0]);
        check::<CargoWorkspaceResponse>(&[0xfa, 0x1d, 0]);
    }

    #[test]
    fn collaborative_message_envelopes_preserve_wire_numbers() {
        fn check<T: EnvelopedMessage + Default + PartialEq>(wire: &[u8]) {
            let message = T::default();
            let envelope = message.clone().into_envelope(0, None, None);
            assert_eq!(envelope.encode_to_vec(), wire, "{}", T::NAME);
            let decoded = Envelope::decode(wire).expect("golden envelope decodes");
            assert_eq!(T::from_envelope(decoded), Some(message));
            assert!(Envelope::decode(&wire[..wire.len() - 1]).is_err());
        }
        check::<OpenCollaborativeChannel>(&[0xa2, 0x1f, 0]);
        check::<OpenCollaborativeChannelResponse>(&[0xaa, 0x1f, 0]);
        check::<CloseCollaborativeChannel>(&[0xb2, 0x1f, 0]);
        check::<GetCollaborativeMessageWindow>(&[0xba, 0x1f, 0]);
        check::<GetCollaborativeMessageWindowResponse>(&[0xc2, 0x1f, 0]);
        check::<ApplyCollaborativeMessageOperation>(&[0xca, 0x1f, 0]);
        check::<ApplyCollaborativeMessageOperationResponse>(&[0xd2, 0x1f, 0]);
        check::<CollaborativeMessageStreamUpdate>(&[0xda, 0x1f, 0]);
        check::<SendChannelMessage>(&[0xca, 0x08, 0]);
        check::<SendChannelMessageResponse>(&[0xd2, 0x08, 0]);
        check::<ChannelMessageSent>(&[0xda, 0x08, 0]);
        check::<GetChannelMessages>(&[0xe2, 0x08, 0]);
        check::<GetChannelMessagesResponse>(&[0xea, 0x08, 0]);
    }

    #[test]
    fn collaborative_stream_status_round_trips_without_message_progress() {
        for (error_code, retry_after_millis) in [(0, 0), (1, 0), (3, 1000)] {
            let update = CollaborativeMessageStreamUpdate {
                contract_version: 1,
                community_id: vec![1; 16],
                channel_id: vec![2; 16],
                status: Some(CollaborativeMessageStreamStatus {
                    error_code,
                    retry_after_millis,
                }),
                ..Default::default()
            };
            let decoded =
                CollaborativeMessageStreamUpdate::decode(update.encode_to_vec().as_slice())
                    .expect("status frame");
            assert_eq!(decoded, update);
            assert_eq!(decoded.outbox_sequence, 0);
            assert!(decoded.message.is_none());
        }
    }

    #[test]
    fn collaborative_message_stream_round_trips_reconciliation_fields() {
        let update = CollaborativeMessageStreamUpdate {
            contract_version: 1,
            community_id: vec![1; 16],
            channel_id: vec![2; 16],
            outbox_sequence: 97,
            message: Some(CollaborativeMessageRecord {
                community_id: vec![1; 16],
                channel_id: vec![2; 16],
                message_id: vec![3; 16],
                source_event_id: vec![4; 32],
                current_event_id: vec![5; 32],
                author_principal_id: vec![6; 16],
                author_display_name: "Ada".into(),
                author_avatar_url: "https://example.test/avatar".into(),
                body: "Updated reply".into(),
                created_at: 1_900_000_000,
                version: 3,
                edited: true,
                deleted: false,
                reply_to_event_id: vec![7; 32],
                reactions: vec![CollaborativeMessageReaction {
                    value: "+".into(),
                    actor_principal_id: vec![8; 16],
                    source_event_id: vec![9; 32],
                }],
                accepted_operation_id: "operation-1".into(),
                outbox_sequence: 97,
                reaction_version: 2,
            }),
            operation_kind: CollaborativeMessageOperationKind::CollaborativeMessageEdit as i32,
            actor_principal_id: vec![6; 16],
            acknowledged_outbox_sequence: 96,
            status: None,
        };
        let envelope =
            update
                .clone()
                .into_envelope(12, Some(11), Some(PeerId { owner_id: 2, id: 3 }));
        let decoded = Envelope::decode(envelope.encode_to_vec().as_slice())
            .expect("populated collaboration envelope");
        assert_eq!(decoded.id, 12);
        assert_eq!(decoded.responding_to, Some(11));
        assert_eq!(decoded.original_sender_id, envelope.original_sender_id);
        assert_eq!(
            CollaborativeMessageStreamUpdate::from_envelope(decoded),
            Some(update)
        );
    }

    #[test]
    fn test_converting_peer_id_from_and_to_u64() {
        let peer_id = PeerId {
            owner_id: 10,
            id: 3,
        };
        assert_eq!(PeerId::from_u64(peer_id.as_u64()), peer_id);
        let peer_id = PeerId {
            owner_id: u32::MAX,
            id: 3,
        };
        assert_eq!(PeerId::from_u64(peer_id.as_u64()), peer_id);
        let peer_id = PeerId {
            owner_id: 10,
            id: u32::MAX,
        };
        assert_eq!(PeerId::from_u64(peer_id.as_u64()), peer_id);
        let peer_id = PeerId {
            owner_id: u32::MAX,
            id: u32::MAX,
        };
        assert_eq!(PeerId::from_u64(peer_id.as_u64()), peer_id);
    }

    #[test]
    fn bounded_buffer_protocol_preserves_admission_limit_and_scan() {
        fn assert_request<
            T: RequestMessage<Response = OpenBoundedBufferResponse> + EntityMessage,
        >() {
        }
        assert_request::<OpenBoundedBufferByPath>();
        let request = OpenBoundedBufferByPath {
            project_id: 7,
            worktree_id: 3,
            path: "target/profile.svg".into(),
            max_bytes: 1024,
        };
        assert_eq!(request.remote_entity_id(), 7);
        let envelope = request.clone().into_envelope(12, None, None);
        let decoded =
            Envelope::decode(envelope.encode_to_vec().as_slice()).expect("bounded buffer envelope");
        assert_eq!(
            OpenBoundedBufferByPath::from_envelope(decoded),
            Some(request)
        );
        let response = OpenBoundedBufferResponse {
            buffer_id: 9,
            worktree_scan_id: 23,
        };
        assert_eq!(
            OpenBoundedBufferResponse::decode(response.encode_to_vec().as_slice())
                .expect("bounded buffer response"),
            response
        );
    }

    #[test]
    fn profile_results_protocol_registers_project_requests_and_normalized_pages() {
        fn assert_page_request<T: RequestMessage<Response = ProfileResultPage> + EntityMessage>() {}
        assert_page_request::<ImportProfileResults>();
        assert_page_request::<GetProfileResultPage>();
        let request = ImportProfileResults {
            project_id: 7,
            import_id: 11,
            protocol_version: 1,
            profile_path: Some(ProjectPath {
                worktree_id: 3,
                path: "target/profile.json".into(),
            }),
            symbols_path: Some(ProjectPath {
                worktree_id: 3,
                path: "target/profile.syms.json".into(),
            }),
            consent_to_transfer: true,
            task_run_id: None,
            artifact_format: "samply-0.13.1".into(),
        };
        assert_eq!(request.remote_entity_id(), 7);
        let envelope = request
            .clone()
            .into_envelope(12, None, Some(PeerId { owner_id: 1, id: 2 }));
        let decoded =
            Envelope::decode(envelope.encode_to_vec().as_slice()).expect("profile envelope");
        assert_eq!(decoded.original_sender_id, envelope.original_sender_id);
        assert_eq!(ImportProfileResults::from_envelope(decoded), Some(request));
        let page = ProfileResultPage {
            import_id: 11,
            generation: 5,
            protocol_version: 1,
            status: ProfileResultStatus::Ready as i32,
            summary: Some(ProfileResultSummary {
                frame_count: 1,
                node_count: 1,
                roots: vec![0],
                total_samples: 2,
                ..Default::default()
            }),
            frames: vec![ProfileResultFrame {
                label: "workload::leaf".into(),
                resolved: true,
                source: Some(ProjectPath {
                    worktree_id: 3,
                    path: "src/lib.rs".into(),
                }),
                source_line: Some(4),
            }],
            nodes: vec![ProfileResultNode {
                inclusive_samples: 2,
                self_samples: 2,
                ..Default::default()
            }],
            complete: true,
            ..Default::default()
        };
        assert_eq!(
            ProfileResultPage::decode(page.encode_to_vec().as_slice()).expect("normalized page"),
            page
        );
        let cancel = CancelProfileImport {
            project_id: 7,
            import_id: 11,
            generation: 5,
        };
        assert_eq!(cancel.remote_entity_id(), 7);
        assert!(!ImportProfileResults::default().consent_to_transfer);
    }

    #[test]
    fn source_coverage_protocol_round_trips_only_project_relative_facts() {
        let response = SourceCoverageResponse {
            project_generation: 3,
            provider_id: "fake-language".to_string(),
            generation: 9,
            status: SourceCoverageStatus::Current as i32,
            files: vec![SourceCoverageFile {
                path: Some(ProjectPath {
                    worktree_id: 7,
                    path: "src/lib.rs".to_string(),
                }),
                ranges: vec![SourceCoverageRange {
                    start_line: 1,
                    start_column: 0,
                    end_line: 1,
                    end_column: 8,
                    hit_count: 4,
                }],
                covered_lines: 1,
                uncovered_lines: 0,
                truncated: false,
            }],
            truncated: false,
            diagnostic: None,
        };
        let bytes = response.encode_to_vec();
        let decoded = SourceCoverageResponse::decode(bytes.as_slice())
            .expect("source coverage response should decode");
        assert_eq!(decoded, response);
        assert!(!String::from_utf8_lossy(&bytes).contains("/Users/"));
    }

    #[test]
    fn test_split_repository_update_keeps_branch_list_on_final_chunk() {
        let update = UpdateRepository {
            updated_statuses: vec![
                StatusEntry::default(),
                StatusEntry::default(),
                StatusEntry::default(),
            ],
            branch_list: vec![Branch {
                ref_name: "refs/heads/main".into(),
                ..Default::default()
            }],
            branch_list_error: Some("partial branch scan".into()),
            ..Default::default()
        };

        let chunks = split_repository_update(update).collect::<Vec<_>>();

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].branch_list.is_empty());
        assert!(chunks[1].branch_list.is_empty());
        assert_eq!(chunks[2].branch_list.len(), 1);
        assert_eq!(chunks[2].branch_list[0].ref_name, "refs/heads/main");
        assert_eq!(chunks[0].branch_list_error, None);
        assert_eq!(chunks[1].branch_list_error, None);
        assert_eq!(
            chunks[2].branch_list_error.as_deref(),
            Some("partial branch scan")
        );
        assert!(!chunks[0].is_last_update);
        assert!(!chunks[1].is_last_update);
        assert!(chunks[2].is_last_update);
    }

    #[test]
    fn cargo_workspace_response_round_trips_project_paths_only() {
        let response = CargoWorkspaceResponse {
            request_id: 41,
            revision: 7,
            input_fingerprint: 11,
            workspaces: vec![CargoWorkspace {
                root: Some(ProjectPath {
                    worktree_id: 3,
                    path: "crates/example".to_string(),
                }),
                root_manifest: Some(ProjectPath {
                    worktree_id: 3,
                    path: "crates/example/Cargo.toml".to_string(),
                }),
                display_name: "example".to_string(),
                is_virtual: false,
                members: Vec::new(),
                dependency_graph: None,
                configuration: Some(CargoWorkspaceConfiguration {
                    profiles: vec![CargoProfile {
                        name: "dev".to_string(),
                        origin: CargoProfileOrigin::Implicit as i32,
                    }],
                    declared_toolchain: None,
                    host_compiler: Some(CargoHostCompiler {
                        status: CargoHostCompilerStatus::Available as i32,
                        release: Some("1.90.0".to_string()),
                        host_target: Some("aarch64-apple-darwin".to_string()),
                        stale: false,
                    }),
                    cargo_target: CargoTargetConfiguration::UnresolvedCargoDefault as i32,
                    diagnostics: Vec::new(),
                    partial: false,
                }),
            }],
            failures: Vec::new(),
            partial: false,
        };
        let encoded = response.encode_to_vec();
        let decoded = CargoWorkspaceResponse::decode(encoded.as_slice())
            .expect("Cargo workspace response should decode");
        assert_eq!(decoded.request_id, 41);
        let root = decoded
            .workspaces
            .first()
            .and_then(|workspace| workspace.root.as_ref())
            .expect("fixture should contain a workspace root");
        assert_eq!(root.path, "crates/example");
        assert!(!String::from_utf8_lossy(&encoded).contains("/Users/"));
        assert!(!String::from_utf8_lossy(&encoded).contains("project_env"));
    }

    #[test]
    fn cargo_dependency_provenance_round_trips_bounded_visible_facts() {
        let dependency = CargoDependency {
            declaration_name: "serde".to_string(),
            rename: Some("serialization".to_string()),
            kind: CargoDependencyKind::Normal as i32,
            version_requirement: "^1".to_string(),
            optional: true,
            uses_default_features: false,
            requested_features: vec!["derive".to_string()],
            target: Some("cfg(unix)".to_string()),
            source_kind: CargoDependencySourceKind::Registry as i32,
            declaration_manifest: Some(ProjectPath {
                worktree_id: 3,
                path: "member/Cargo.toml".to_string(),
            }),
            declaration_origin: CargoDependencyDeclarationOrigin::WorkspaceInherited as i32,
            resolved_instances: vec![CargoResolvedDependency {
                name: "serde".to_string(),
                version: "1.0.0".to_string(),
                source_kind: CargoDependencySourceKind::Registry as i32,
                enabled_features: vec!["derive".to_string()],
                lock_status: CargoDependencyLockStatus::Locked as i32,
                ..Default::default()
            }],
            feature_causality: CargoDependencyFeatureCausality::Validated as i32,
            ..Default::default()
        };
        let encoded = dependency.encode_to_vec();
        let decoded = CargoDependency::decode(encoded.as_slice())
            .expect("Cargo dependency provenance should decode");
        assert_eq!(decoded.resolved_instances.len(), 1);
        assert_eq!(
            decoded
                .declaration_manifest
                .as_ref()
                .map(|path| path.path.as_str()),
            Some("member/Cargo.toml")
        );
        let wire = String::from_utf8_lossy(&encoded);
        assert!(!wire.contains("/Users/"));
        assert!(!wire.contains(".cargo/registry"));
    }

    #[test]
    fn update_project_upstream_session_preserves_legacy_absence() {
        let legacy = UpdateProject::decode(&[8, 7][..]).expect("legacy project metadata");
        assert_eq!(legacy.project_id, 7);
        assert_eq!(legacy.upstream_session_generation, None);
        let explicit_zero =
            UpdateProject::decode(&[8, 7, 24, 0][..]).expect("explicit session zero");
        assert_eq!(explicit_zero.upstream_session_generation, Some(0));
        let replacement =
            UpdateProject::decode(&[8, 7, 24, 1][..]).expect("replacement project session");
        assert_eq!(replacement.upstream_session_generation, Some(1));
        assert_eq!(replacement.encode_to_vec(), vec![8, 7, 24, 1]);
    }

    #[test]
    fn structured_node_host_experimental_enabled_preserves_legacy_absence() {
        let legacy = [10, 6, b'c', b'a', b's', b'e', b'-', b'a', 32, 4];
        let node = StructuredExecutionNode::decode(legacy.as_slice()).expect("legacy node");
        assert_eq!(node.node_id, "case-a");
        assert_eq!(node.experimental_feature, None);
        assert_eq!(node.experimental_feature_enabled, None);
        assert_eq!(node.encode_to_vec(), legacy);
        for (value, encoded_value) in [(false, 0), (true, 1)] {
            let mut encoded = legacy.to_vec();
            encoded.extend([64, encoded_value]);
            let node = StructuredExecutionNode::decode(encoded.as_slice())
                .expect("explicit optional host DATA");
            assert_eq!(node.experimental_feature_enabled, Some(value));
            assert_eq!(node.encode_to_vec(), encoded);
        }
    }

    #[test]
    fn structured_execution_protocol_is_bounded_and_path_safe() {
        let page = StructuredExecutionSnapshotPage {
            protocol_version: 1,
            project_generation: 7,
            provider_id: "web-tests".to_string(),
            discovery_generation: 3,
            status: StructuredProviderStatus::Current as i32,
            nodes: vec![StructuredExecutionNode {
                experimental_feature: None,
                experimental_feature_enabled: None,
                debug_capability: None,
                node_id: "case-a".to_string(),
                parent_id: Some("suite".to_string()),
                label: "renders a page".to_string(),
                kind: StructuredExecutionNodeKind::Case as i32,
                path: Some(ProjectPath {
                    worktree_id: 2,
                    path: "web/tests.js".to_string(),
                }),
            }],
            next_page_start: 0,
            partial: false,
            diagnostic: None,
            current_run: None,
            last_complete_run: None,
        };
        let encoded = page.encode_to_vec();
        let decoded = StructuredExecutionSnapshotPage::decode(encoded.as_slice())
            .expect("structured execution page should decode");
        assert_eq!(decoded.nodes.len(), 1);
        assert_eq!(
            decoded.nodes[0]
                .path
                .as_ref()
                .map(|path| path.path.as_str()),
            Some("web/tests.js")
        );
        let wire = String::from_utf8_lossy(&encoded);
        assert!(!wire.contains("/Users/"));
        assert!(!wire.contains("project_env"));
        assert!(!wire.contains("terminal_bytes"));
    }
}
