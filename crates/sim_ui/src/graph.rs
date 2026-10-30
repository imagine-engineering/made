use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use sim_runtime::{
    ContentRevision, GraphCommand, GraphCommandEngine, GraphDocument, GraphError, GraphIdentifier,
    GraphPoint, GraphSelection, GraphViewport, MAX_GRAPH_SNAPSHOT_BYTES, MAX_WORKFLOW_BYTES,
    ProfileId, WorkflowAuthority, WorkflowSaveCoordinator, WorkflowSaveError,
    WorkflowStorageProvider,
};
use thiserror::Error;

pub const GRAPH_WORKSPACE_SCHEMA_VERSION: u16 = 3;
pub const MAX_GRAPH_WORKSPACE_SNAPSHOT_BYTES: usize = 608 * 1024 * 1024;
pub const MAX_GRAPH_OPERATION_ERRORS: usize = 64;
const MAX_LEGACY_GRAPH_WORKSPACE_SNAPSHOT_BYTES: usize = 72 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub enum WorkflowOpenState {
    Editable(GraphCommandEngine),
    ReadOnly {
        original_bytes: Vec<u8>,
        diagnostic: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphWorkspaceModel {
    pub schema_version: u16,
    pub title: String,
    pub open_state: WorkflowOpenState,
    pub save_coordinator: WorkflowSaveCoordinator,
    pub execution_association: Option<String>,
    pub canvas_info_visible: bool,
    pub last_error: Option<String>,
    pub operation_errors: Vec<String>,
    pub announcement: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedGraphWorkspaceV3 {
    schema_version: u16,
    title: String,
    engine: Option<String>,
    read_only_bytes: Option<String>,
    read_only_diagnostic: Option<String>,
    save_journal: String,
    execution_association: Option<String>,
    canvas_info_visible: bool,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    operation_errors: Vec<String>,
    #[serde(default)]
    announcement: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedGraphWorkspaceLegacy {
    schema_version: u16,
    title: String,
    engine: Option<Vec<u8>>,
    read_only_bytes: Option<Vec<u8>>,
    read_only_diagnostic: Option<String>,
    save_journal: Vec<u8>,
    execution_association: Option<String>,
    canvas_info_visible: bool,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    operation_errors: Vec<String>,
    #[serde(default)]
    announcement: Option<String>,
}

#[derive(Deserialize)]
struct PersistedGraphWorkspaceHeader {
    schema_version: u16,
}

struct DecodedGraphWorkspace {
    title: String,
    engine: Option<Vec<u8>>,
    read_only_bytes: Option<Vec<u8>>,
    read_only_diagnostic: Option<String>,
    save_journal: Vec<u8>,
    execution_association: Option<String>,
    canvas_info_visible: bool,
    last_error: Option<String>,
    operation_errors: Vec<String>,
    announcement: Option<String>,
}

fn decode_snapshot_field(
    field: &'static str,
    encoded: Option<String>,
    maximum_decoded_bytes: usize,
) -> Result<Option<Vec<u8>>, GraphWorkspaceError> {
    encoded
        .map(|encoded| decode_required_snapshot_field(field, encoded, maximum_decoded_bytes))
        .transpose()
}

fn decode_required_snapshot_field(
    field: &'static str,
    encoded: String,
    maximum_decoded_bytes: usize,
) -> Result<Vec<u8>, GraphWorkspaceError> {
    let maximum_encoded_bytes = maximum_decoded_bytes
        .checked_add(2)
        .and_then(|length| length.checked_div(3))
        .and_then(|length| length.checked_mul(4))
        .ok_or_else(|| {
            GraphWorkspaceError::Persistence(format!(
                "could not calculate the encoded size limit for {field}"
            ))
        })?;
    if encoded.len() > maximum_encoded_bytes {
        return Err(GraphWorkspaceError::SnapshotFieldTooLarge(
            field,
            encoded.len(),
            maximum_encoded_bytes,
        ));
    }
    let decoded = BASE64
        .decode(encoded)
        .map_err(|error| GraphWorkspaceError::Persistence(format!("invalid {field}: {error}")))?;
    if decoded.len() > maximum_decoded_bytes {
        return Err(GraphWorkspaceError::SnapshotFieldTooLarge(
            field,
            decoded.len(),
            maximum_decoded_bytes,
        ));
    }
    Ok(decoded)
}

impl GraphWorkspaceModel {
    pub fn create(title: impl Into<String>) -> Result<Self, GraphWorkspaceError> {
        let document = GraphDocument::default();
        let identity = document.document_identity.to_string();
        let engine = GraphCommandEngine::new(document)?;
        let bytes = engine.document.to_workflow_bytes()?;
        let save_coordinator =
            WorkflowSaveCoordinator::new(identity, WorkflowStorageProvider::Draft, bytes)?;
        Ok(Self {
            schema_version: GRAPH_WORKSPACE_SCHEMA_VERSION,
            title: title.into(),
            open_state: WorkflowOpenState::Editable(engine),
            save_coordinator,
            execution_association: None,
            canvas_info_visible: false,
            last_error: None,
            operation_errors: Vec::new(),
            announcement: Some("Created native workflow".to_owned()),
        })
    }

    pub fn open(
        title: impl Into<String>,
        document_identity: impl Into<String>,
        provider: WorkflowStorageProvider,
        bytes: Vec<u8>,
    ) -> Result<Self, GraphWorkspaceError> {
        let save_coordinator =
            WorkflowSaveCoordinator::new(document_identity, provider, bytes.clone())?;
        let open_state =
            match GraphDocument::from_workflow_bytes(&bytes).and_then(GraphCommandEngine::new) {
                Ok(engine) => WorkflowOpenState::Editable(engine),
                Err(error) => WorkflowOpenState::ReadOnly {
                    original_bytes: bytes,
                    diagnostic: error.to_string(),
                },
            };
        let announcement = match &open_state {
            WorkflowOpenState::Editable(_) => "Opened native workflow".to_owned(),
            WorkflowOpenState::ReadOnly { diagnostic, .. } => {
                format!("Opened workflow read-only: {diagnostic}")
            }
        };
        Ok(Self {
            schema_version: GRAPH_WORKSPACE_SCHEMA_VERSION,
            title: title.into(),
            open_state,
            save_coordinator,
            execution_association: None,
            canvas_info_visible: false,
            last_error: None,
            operation_errors: Vec::new(),
            announcement: Some(announcement),
        })
    }

    pub fn engine(&self) -> Option<&GraphCommandEngine> {
        match &self.open_state {
            WorkflowOpenState::Editable(engine) => Some(engine),
            WorkflowOpenState::ReadOnly { .. } => None,
        }
    }

    pub fn document(&self) -> Option<&GraphDocument> {
        self.engine().map(|engine| &engine.document)
    }

    pub fn bind_profile_identity(&mut self, profile_id: ProfileId) {
        if let WorkflowOpenState::Editable(engine) = &mut self.open_state {
            engine.bind_profile_identity(profile_id.0);
        }
    }

    pub fn replace_ephemeral_graph_state(
        &mut self,
        selection: GraphSelection,
        viewport: GraphViewport,
    ) -> Result<(), GraphWorkspaceError> {
        let engine = match &mut self.open_state {
            WorkflowOpenState::Editable(engine) => engine,
            WorkflowOpenState::ReadOnly { .. } => return Err(GraphWorkspaceError::ReadOnly),
        };
        engine.replace_workspace_state(selection, viewport)?;
        Ok(())
    }

    pub fn is_read_only(&self) -> bool {
        matches!(self.open_state, WorkflowOpenState::ReadOnly { .. })
    }

    pub fn read_only_diagnostic(&self) -> Option<&str> {
        match &self.open_state {
            WorkflowOpenState::ReadOnly { diagnostic, .. } => Some(diagnostic),
            WorkflowOpenState::Editable(_) => None,
        }
    }

    pub fn original_bytes(&self) -> &[u8] {
        match &self.open_state {
            WorkflowOpenState::Editable(_) => &self.save_coordinator.base().bytes,
            WorkflowOpenState::ReadOnly { original_bytes, .. } => original_bytes,
        }
    }

    pub fn apply(&mut self, command: GraphCommand) -> Result<(), GraphWorkspaceError> {
        let result = self.try_apply_with_change(command).map(drop);
        self.record_result(&result, "Graph edit applied");
        result
    }

    pub fn apply_with_change(
        &mut self,
        command: GraphCommand,
    ) -> Result<bool, GraphWorkspaceError> {
        let result = self.try_apply_with_change(command);
        self.record_result(&result, "Graph edit applied");
        result
    }

    fn try_apply_with_change(
        &mut self,
        command: GraphCommand,
    ) -> Result<bool, GraphWorkspaceError> {
        let WorkflowOpenState::Editable(engine) = &self.open_state else {
            return Err(GraphWorkspaceError::ReadOnly);
        };
        let mut candidate_engine = engine.clone();
        candidate_engine.apply(command)?;
        if candidate_engine.document == engine.document {
            return Ok(false);
        }
        let bytes = candidate_engine.document.to_workflow_bytes()?;
        let mut candidate_save = self.save_coordinator.clone();
        candidate_save.edit(bytes)?;
        self.open_state = WorkflowOpenState::Editable(candidate_engine);
        self.save_coordinator = candidate_save;
        Ok(true)
    }

    pub fn undo(&mut self) -> Result<bool, GraphWorkspaceError> {
        self.apply_history(false)
    }

    pub fn redo(&mut self) -> Result<bool, GraphWorkspaceError> {
        self.apply_history(true)
    }

    fn apply_history(&mut self, redo: bool) -> Result<bool, GraphWorkspaceError> {
        let result = (|| {
            let WorkflowOpenState::Editable(engine) = &self.open_state else {
                return Err(GraphWorkspaceError::ReadOnly);
            };
            let mut candidate_engine = engine.clone();
            let changed = if redo {
                candidate_engine.redo()
            } else {
                candidate_engine.undo()
            };
            if !changed {
                return Ok(false);
            }
            let bytes = candidate_engine.document.to_workflow_bytes()?;
            let mut candidate_save = self.save_coordinator.clone();
            candidate_save.edit(bytes)?;
            self.open_state = WorkflowOpenState::Editable(candidate_engine);
            self.save_coordinator = candidate_save;
            Ok(true)
        })();
        self.record_result(
            &result,
            if redo {
                "Graph edit redone"
            } else {
                "Graph edit undone"
            },
        );
        result
    }

    pub fn observe_external_change(&mut self, bytes: Vec<u8>) -> Result<(), GraphWorkspaceError> {
        let result = self
            .save_coordinator
            .observe_external_change(bytes)
            .map_err(GraphWorkspaceError::from);
        self.record_result(&result, "External workflow change detected");
        result
    }

    pub fn observe_external_deletion(&mut self) -> Result<(), GraphWorkspaceError> {
        let result = self
            .save_coordinator
            .observe_external_deletion()
            .map_err(GraphWorkspaceError::from);
        self.record_result(&result, "External workflow deletion detected");
        result
    }

    pub fn reload_external(&mut self) -> Result<(), GraphWorkspaceError> {
        let result = (|| {
            let mut coordinator = self.save_coordinator.clone();
            coordinator.reload_external()?;
            let bytes = coordinator.local_bytes().to_vec();
            let open_state = match GraphDocument::from_workflow_bytes(&bytes)
                .and_then(GraphCommandEngine::new)
            {
                Ok(engine) => WorkflowOpenState::Editable(engine),
                Err(error) => WorkflowOpenState::ReadOnly {
                    original_bytes: bytes,
                    diagnostic: error.to_string(),
                },
            };
            self.open_state = open_state;
            self.save_coordinator = coordinator;
            Ok(())
        })();
        self.record_result(&result, "External workflow version reloaded");
        result
    }

    pub fn reload_from_storage(&mut self, bytes: Vec<u8>) -> Result<(), GraphWorkspaceError> {
        let result = (|| {
            let document_identity = self.save_coordinator.document_identity().to_owned();
            let provider = self.save_coordinator.provider().clone();
            let save_coordinator =
                WorkflowSaveCoordinator::new(document_identity, provider, bytes.clone())?;
            let open_state = match GraphDocument::from_workflow_bytes(&bytes)
                .and_then(GraphCommandEngine::new)
            {
                Ok(engine) => WorkflowOpenState::Editable(engine),
                Err(error) => WorkflowOpenState::ReadOnly {
                    original_bytes: bytes,
                    diagnostic: error.to_string(),
                },
            };
            self.open_state = open_state;
            self.save_coordinator = save_coordinator;
            Ok(())
        })();
        match &result {
            Ok(()) => {
                self.last_error = None;
                self.announcement = Some(match &self.open_state {
                    WorkflowOpenState::Editable(_) => "Workflow reloaded from storage".to_owned(),
                    WorkflowOpenState::ReadOnly { diagnostic, .. } => {
                        format!("Workflow reloaded read-only: {diagnostic}")
                    }
                });
            }
            Err(error) => self.report_error(error),
        }
        result
    }

    pub fn keep_local(&mut self) -> Result<(), GraphWorkspaceError> {
        let result = self
            .save_coordinator
            .keep_local()
            .map_err(GraphWorkspaceError::from);
        self.record_result(&result, "Kept local workflow version");
        result
    }

    pub fn prepare_save(
        &mut self,
        operation_id: uuid::Uuid,
        observed_revision: ContentRevision,
        target_identity: impl Into<String>,
        save_copy: bool,
    ) -> Result<sim_runtime::PreparedWorkflowSave, GraphWorkspaceError> {
        let result = self
            .save_coordinator
            .prepare_save(operation_id, observed_revision, target_identity, save_copy)
            .map_err(GraphWorkspaceError::from);
        self.record_result(&result, "Workflow save prepared");
        result
    }

    pub fn commit_save(
        &mut self,
        operation_id: uuid::Uuid,
        observed_revision: ContentRevision,
        committed_revision: ContentRevision,
    ) -> Result<(), GraphWorkspaceError> {
        let result = self
            .save_coordinator
            .commit_save(operation_id, observed_revision, committed_revision)
            .map_err(GraphWorkspaceError::from);
        self.record_result(&result, "Workflow save committed");
        result
    }

    pub fn commit_draft_save(&mut self) -> Result<(), GraphWorkspaceError> {
        let operation_id = uuid::Uuid::new_v4();
        let observed_revision = self.save_coordinator.base().revision.clone();
        let target_identity = self.save_coordinator.document_identity().to_owned();
        let prepared = self.prepare_save(
            operation_id,
            observed_revision.clone(),
            target_identity,
            false,
        )?;
        let committed_revision = ContentRevision::from_bytes(&prepared.bytes);
        self.commit_save(operation_id, observed_revision, committed_revision)
    }

    pub fn recover_after_restart(&mut self) {
        self.save_coordinator.recover_after_restart();
        self.announcement = Some(if let Some(error) = &self.last_error {
            format!("Workflow state restored with prior failure: {error}")
        } else {
            match self.save_coordinator.authority() {
                WorkflowAuthority::Interrupted => "Interrupted workflow save recovered".to_owned(),
                _ => "Workflow state restored".to_owned(),
            }
        });
    }

    pub fn encode(&self) -> Result<Vec<u8>, GraphWorkspaceError> {
        let (engine, read_only_bytes, read_only_diagnostic) = match &self.open_state {
            WorkflowOpenState::Editable(engine) => {
                (Some(BASE64.encode(engine.encode()?)), None, None)
            }
            WorkflowOpenState::ReadOnly {
                original_bytes,
                diagnostic,
            } => (
                None,
                Some(BASE64.encode(original_bytes)),
                Some(diagnostic.clone()),
            ),
        };
        let persisted = PersistedGraphWorkspaceV3 {
            schema_version: GRAPH_WORKSPACE_SCHEMA_VERSION,
            title: self.title.clone(),
            engine,
            read_only_bytes,
            read_only_diagnostic,
            save_journal: BASE64.encode(self.save_coordinator.encode()?),
            execution_association: self.execution_association.clone(),
            canvas_info_visible: self.canvas_info_visible,
            last_error: self.last_error.clone(),
            operation_errors: self.operation_errors.clone(),
            announcement: self.announcement.clone(),
        };
        let bytes = serde_json::to_vec(&persisted)
            .map_err(|error| GraphWorkspaceError::Persistence(error.to_string()))?;
        if bytes.len() > MAX_GRAPH_WORKSPACE_SNAPSHOT_BYTES {
            return Err(GraphWorkspaceError::SnapshotTooLarge(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, GraphWorkspaceError> {
        if bytes.len() > MAX_GRAPH_WORKSPACE_SNAPSHOT_BYTES {
            return Err(GraphWorkspaceError::SnapshotTooLarge(bytes.len()));
        }
        let header: PersistedGraphWorkspaceHeader = serde_json::from_slice(bytes)
            .map_err(|error| GraphWorkspaceError::Persistence(error.to_string()))?;
        let persisted = match header.schema_version {
            GRAPH_WORKSPACE_SCHEMA_VERSION => {
                let persisted: PersistedGraphWorkspaceV3 = serde_json::from_slice(bytes)
                    .map_err(|error| GraphWorkspaceError::Persistence(error.to_string()))?;
                DecodedGraphWorkspace {
                    title: persisted.title,
                    engine: decode_snapshot_field(
                        "engine",
                        persisted.engine,
                        MAX_GRAPH_SNAPSHOT_BYTES,
                    )?,
                    read_only_bytes: decode_snapshot_field(
                        "read_only_bytes",
                        persisted.read_only_bytes,
                        MAX_WORKFLOW_BYTES,
                    )?,
                    read_only_diagnostic: persisted.read_only_diagnostic,
                    save_journal: decode_required_snapshot_field(
                        "save_journal",
                        persisted.save_journal,
                        MAX_GRAPH_WORKSPACE_SNAPSHOT_BYTES,
                    )?,
                    execution_association: persisted.execution_association,
                    canvas_info_visible: persisted.canvas_info_visible,
                    last_error: persisted.last_error,
                    operation_errors: persisted.operation_errors,
                    announcement: persisted.announcement,
                }
            }
            1 | 2 => {
                if bytes.len() > MAX_LEGACY_GRAPH_WORKSPACE_SNAPSHOT_BYTES {
                    return Err(GraphWorkspaceError::SnapshotTooLarge(bytes.len()));
                }
                let persisted: PersistedGraphWorkspaceLegacy = serde_json::from_slice(bytes)
                    .map_err(|error| GraphWorkspaceError::Persistence(error.to_string()))?;
                DecodedGraphWorkspace {
                    title: persisted.title,
                    engine: persisted.engine,
                    read_only_bytes: persisted.read_only_bytes,
                    read_only_diagnostic: persisted.read_only_diagnostic,
                    save_journal: persisted.save_journal,
                    execution_association: persisted.execution_association,
                    canvas_info_visible: persisted.canvas_info_visible,
                    last_error: persisted.last_error,
                    operation_errors: persisted.operation_errors,
                    announcement: persisted.announcement,
                }
            }
            schema_version => return Err(GraphWorkspaceError::UnsupportedSchema(schema_version)),
        };
        let save_coordinator = WorkflowSaveCoordinator::decode(&persisted.save_journal)?;
        let open_state = match (
            persisted.engine,
            persisted.read_only_bytes,
            persisted.read_only_diagnostic,
        ) {
            (Some(engine), None, None) => {
                WorkflowOpenState::Editable(GraphCommandEngine::decode(&engine)?)
            }
            (None, Some(original_bytes), Some(diagnostic)) => WorkflowOpenState::ReadOnly {
                original_bytes,
                diagnostic,
            },
            _ => return Err(GraphWorkspaceError::InvalidSnapshotState),
        };
        match &open_state {
            WorkflowOpenState::Editable(engine) => {
                let saved_document =
                    GraphDocument::from_workflow_bytes(save_coordinator.local_bytes())
                        .map_err(|_| GraphWorkspaceError::InvalidSnapshotState)?;
                if !engine
                    .document
                    .has_same_persisted_workflow(&saved_document)?
                {
                    return Err(GraphWorkspaceError::InvalidSnapshotState);
                }
            }
            WorkflowOpenState::ReadOnly { original_bytes, .. } => {
                if original_bytes != save_coordinator.local_bytes() {
                    return Err(GraphWorkspaceError::InvalidSnapshotState);
                }
            }
        }
        let mut operation_errors = persisted.operation_errors;
        if operation_errors.len() > MAX_GRAPH_OPERATION_ERRORS {
            operation_errors.drain(..operation_errors.len() - MAX_GRAPH_OPERATION_ERRORS);
        }
        let mut model = Self {
            schema_version: GRAPH_WORKSPACE_SCHEMA_VERSION,
            title: persisted.title,
            open_state,
            save_coordinator,
            execution_association: persisted.execution_association,
            canvas_info_visible: persisted.canvas_info_visible,
            last_error: persisted.last_error,
            operation_errors,
            announcement: persisted.announcement,
        };
        model.recover_after_restart();
        Ok(model)
    }

    pub fn selection(&self) -> Option<&GraphSelection> {
        self.document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| &graph.selection)
    }

    pub fn selected_node_identifiers(&self) -> Vec<GraphIdentifier> {
        self.selection()
            .map(|selection| selection.nodes.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn graph_point_for_paste(&self) -> GraphPoint {
        self.document()
            .and_then(|document| document.active_graph().ok())
            .map(|graph| {
                graph
                    .viewport
                    .screen_to_graph(GraphPoint { x: 80.0, y: 80.0 })
            })
            .unwrap_or(GraphPoint::ZERO)
    }

    pub fn report_error(&mut self, error: impl std::fmt::Display) {
        self.record_failure(error.to_string());
    }

    fn record_result<T, E: std::fmt::Display>(&mut self, result: &Result<T, E>, success: &str) {
        match result {
            Ok(_) => {
                self.last_error = None;
                self.announcement = Some(success.to_owned());
            }
            Err(error) => {
                self.record_failure(error.to_string());
            }
        }
    }

    fn record_failure(&mut self, error: String) {
        if self.operation_errors.last() != Some(&error) {
            if self.operation_errors.len() == MAX_GRAPH_OPERATION_ERRORS {
                self.operation_errors.drain(..1);
            }
            self.operation_errors.push(error.clone());
        }
        self.last_error = Some(error.clone());
        self.announcement = Some(format!("Graph operation failed: {error}"));
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum GraphWorkspaceError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Save(#[from] WorkflowSaveError),
    #[error("workflow is open read-only")]
    ReadOnly,
    #[error("graph workspace schema {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("graph workspace snapshot has inconsistent editable/read-only fields")]
    InvalidSnapshotState,
    #[error("graph workspace snapshot is {0} bytes, exceeding its limit")]
    SnapshotTooLarge(usize),
    #[error("graph workspace snapshot field {0} is {1} bytes, exceeding its {2}-byte limit")]
    SnapshotFieldTooLarge(&'static str, usize, usize),
    #[error("graph workspace persistence failed: {0}")]
    Persistence(String),
}
