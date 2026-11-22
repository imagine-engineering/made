use std::{collections::BTreeSet, panic::AssertUnwindSafe, sync::Arc};

use graphite_core_types::registry::NODE_REGISTRY;
use graphite_editor::messages::portfolio::document::{
    DocumentMessageHandler, utility_types::error::EditorError,
};
use graphite_graph_craft::document::DocumentNodeImplementation;

use super::OriginalJson;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyDocumentSchema {
    NetworkInterface,
    OriginalNetwork,
}

#[derive(Debug, thiserror::Error)]
pub enum LegacyInspectionError {
    #[error("The Graphite document exceeds its input byte budget.")]
    BudgetExceeded,
    #[error("The Graphite document does not identify a supported legacy layout.")]
    UnsupportedLayout,
    #[error("The Graphite document is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("The Graphite document JSON could not be read: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Source(#[from] EditorError),
    #[error("The original Graphite document loader panicked.")]
    SourcePanic,
    #[error("The local Graphite node registry is unavailable.")]
    RegistryUnavailable,
}

pub struct PreservedLegacyDocument {
    original_bytes: Arc<[u8]>,
    schema: LegacyDocumentSchema,
    build_hint: Option<String>,
    node_count: usize,
    unregistered_identifiers: BTreeSet<String>,
}

impl PreservedLegacyDocument {
    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn schema(&self) -> LegacyDocumentSchema {
        self.schema
    }

    pub fn build_hint(&self) -> Option<&str> {
        self.build_hint.as_deref()
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn unregistered_identifiers(&self) -> &BTreeSet<String> {
        &self.unregistered_identifiers
    }
}

pub fn inspect_legacy_document(
    bytes: &Arc<[u8]>,
    max_input_bytes: usize,
) -> Result<PreservedLegacyDocument, LegacyInspectionError> {
    if bytes.len() > max_input_bytes {
        return Err(LegacyInspectionError::BudgetExceeded);
    }
    let string = std::str::from_utf8(bytes)?;
    let OriginalJson(value) = serde_json::from_str(string)?;
    let fields = value
        .as_object()
        .ok_or(LegacyInspectionError::UnsupportedLayout)?;
    let schema = match (fields.get("network_interface"), fields.get("network")) {
        (Some(interface), None) if interface.is_object() => LegacyDocumentSchema::NetworkInterface,
        (None, Some(network)) if network.is_object() => LegacyDocumentSchema::OriginalNetwork,
        _ => return Err(LegacyInspectionError::UnsupportedLayout),
    };
    let build_hint = fields
        .get("commit_hash")
        .and_then(|value| value.as_str())
        .map(str::to_owned);
    drop(value);
    let document = std::panic::catch_unwind(AssertUnwindSafe(|| {
        DocumentMessageHandler::deserialize_document(string)
    }))
    .map_err(|_| LegacyInspectionError::SourcePanic)??;
    let registry = NODE_REGISTRY
        .lock()
        .map_err(|_| LegacyInspectionError::RegistryUnavailable)?;
    if registry.is_empty() {
        return Err(LegacyInspectionError::RegistryUnavailable);
    }

    let mut node_count = 0;
    let mut unregistered_identifiers = BTreeSet::new();
    let mut networks = vec![document.document_network()];
    while let Some(network) = networks.pop() {
        node_count += network.nodes.len();
        for node in network.nodes.values() {
            match &node.implementation {
                DocumentNodeImplementation::Network(network) => networks.push(network),
                DocumentNodeImplementation::ProtoNode(identifier)
                    if !registry.contains_key(identifier) =>
                {
                    unregistered_identifiers.insert(identifier.as_str().to_owned());
                }
                _ => {}
            }
        }
    }
    Ok(PreservedLegacyDocument {
        original_bytes: bytes.clone(),
        schema,
        build_hint,
        node_count,
        unregistered_identifiers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphite_core_types::ProtoNodeIdentifier;
    use graphite_editor::messages::portfolio::document::utility_types::network_interface::{
        NodeTemplate, NodeTemplateImplementation,
    };
    use graphite_graph_craft::document::NodeId;

    #[test]
    fn at_r06_1_inspects_pinned_legacy_and_preserves_unknown_source_fields() {
        let mut owner = DocumentMessageHandler::default();
        let identifier = "future_migration_fixture::UnknownNode";
        assert!(
            owner
                .network_interface
                .insert_node(
                    NodeId(81),
                    NodeTemplate {
                        implementation: NodeTemplateImplementation::ProtoNode(
                            ProtoNodeIdentifier::new(identifier)
                        ),
                        ..Default::default()
                    },
                    &[],
                )
                .is_none()
        );
        owner.commit_hash = "original-source-build".into();
        let mut value = serde_json::to_value(&owner).expect("pinned Graphite serializer");
        value["future_metadata"] = serde_json::json!({"name": "保存", "opaque": [0, 19, 82]});
        let mut raw = serde_json::to_vec_pretty(&value).expect("valid source JSON");
        raw.push(b'\n');
        let bytes = Arc::<[u8]>::from(raw.as_slice());
        let inspected =
            inspect_legacy_document(&bytes, 1024 * 1024).expect("open the pinned source layout");
        assert_eq!(inspected.schema(), LegacyDocumentSchema::NetworkInterface);
        assert_eq!(inspected.node_count(), owner.document_network().nodes.len());
        assert_eq!(inspected.build_hint(), Some("original-source-build"));
        assert!(inspected.unregistered_identifiers().contains(identifier));
        assert_eq!(inspected.original_bytes(), raw);
    }

    #[test]
    fn at_r06_1_reports_duplicate_unreadable_and_unrecognized_legacy_layouts() {
        let duplicate =
            Arc::<[u8]>::from(br#"{"network_interface":{},"network_interface":{}}"#.as_slice());
        assert!(
            matches!(inspect_legacy_document(&duplicate, 4096), Err(LegacyInspectionError::Json(error)) if error.to_string().contains("duplicate object key"))
        );
        let future =
            Arc::<[u8]>::from(br#"{"format_version":2,"graph":{},"field":[0,1]}"#.as_slice());
        assert!(matches!(
            inspect_legacy_document(&future, 4096),
            Err(LegacyInspectionError::UnsupportedLayout)
        ));
        let incomplete = Arc::<[u8]>::from(br#"{"network_interface":{}}"#.as_slice());
        assert!(matches!(
            inspect_legacy_document(&incomplete, 4096),
            Err(LegacyInspectionError::Source(_))
        ));
        assert!(matches!(
            inspect_legacy_document(&future, 1),
            Err(LegacyInspectionError::BudgetExceeded)
        ));
    }
}
