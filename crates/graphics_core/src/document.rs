use std::{fmt, panic::AssertUnwindSafe};

use futures::FutureExt as _;

use graphite_core_types::NodeParameter;
use graphite_graph_craft::document::{
    DocumentNode, DocumentNodeImplementation, NodeId, NodeInput, NodeNetwork, value::TaggedValue,
};
use graphite_math_nodes::add;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericAddInput {
    Augend,
    Addend,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericAddCommand {
    /// The caller's original editor DocumentId.0 representation, not a minted catalog identity.
    pub document_id_encoding: u64,
    pub network_path: Vec<NodeId>,
    pub target_node_id: NodeId,
    pub expected_revision: u64,
    pub input: NumericAddInput,
    pub value: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NumericDocumentError {
    ForeignDocument,
    UnsupportedNetworkPath,
    MissingNode,
    StaleRevision { expected: u64, actual: u64 },
    InvalidNodeContract,
    InvalidInputType,
    NonFiniteValue,
    RevisionExhausted,
}

impl fmt::Display for NumericDocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignDocument => {
                formatter.write_str("The command belongs to another document.")
            }
            Self::UnsupportedNetworkPath => {
                formatter.write_str("Only the root network is admitted.")
            }
            Self::MissingNode => formatter.write_str("The addressed Add node does not exist."),
            Self::StaleRevision { expected, actual } => {
                write!(
                    formatter,
                    "The command expects revision {expected}, but the document is at {actual}."
                )
            }
            Self::InvalidNodeContract => {
                formatter.write_str("The network is outside the numeric Add contract.")
            }
            Self::InvalidInputType => {
                formatter.write_str("The Add input must contain an original Number value.")
            }
            Self::NonFiniteValue => {
                formatter.write_str("The numeric document accepts finite inputs only.")
            }
            Self::RevisionExhausted => formatter.write_str("The document revision is exhausted."),
        }
    }
}

impl std::error::Error for NumericDocumentError {}

#[derive(Debug)]
pub enum NumericNetworkEvaluationError {
    Validation(NumericDocumentError),
    Compilation(String),
    Construction(graphite_graph_craft::proto::GraphErrors),
    Execution(Box<dyn std::error::Error>),
    UnexpectedOutput(Box<TaggedValue>),
    SourcePanic(String),
}

impl fmt::Display for NumericNetworkEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(error) => error.fmt(formatter),
            Self::Compilation(error) => {
                write!(formatter, "Original graph compilation failed: {error}")
            }
            Self::Construction(errors) => {
                formatter.write_str("Original graph construction failed:")?;
                for error in errors {
                    write!(formatter, " {error:?};")?;
                }
                Ok(())
            }
            Self::Execution(error) => write!(formatter, "Original graph execution failed: {error}"),
            Self::SourcePanic(detail) => {
                write!(formatter, "Original numeric graph panicked: {detail}")
            }
            Self::UnexpectedOutput(output) => write!(
                formatter,
                "Original numeric graph returned an unexpected value: {output:?}"
            ),
        }
    }
}

impl std::error::Error for NumericNetworkEvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::Execution(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

pub struct NumericAddDocument {
    document_id_encoding: u64,
    node_id: NodeId,
    revision: u64,
    network: NodeNetwork,
}

impl NumericAddDocument {
    /// Preserves the caller's editor DocumentId.0 u64 encoding without importing the editor owner.
    /// This process-local subset does not validate durable catalog identity or persisted ownership.
    pub fn new(
        document_id_encoding: u64,
        node_id: NodeId,
        augend: f64,
        addend: f64,
    ) -> Result<Self, NumericDocumentError> {
        if !augend.is_finite() || !addend.is_finite() {
            return Err(NumericDocumentError::NonFiniteValue);
        }
        let node = DocumentNode {
            inputs: vec![
                NodeInput::value(TaggedValue::Number(augend), true),
                NodeInput::value(TaggedValue::Number(addend), true),
            ],
            implementation: DocumentNodeImplementation::ProtoNode(add::IDENTIFIER),
            ..DocumentNode::default()
        };
        Ok(Self {
            document_id_encoding,
            node_id,
            revision: 0,
            network: NodeNetwork {
                nodes: [(node_id, node)].into_iter().collect(),
                exports: vec![NodeInput::node(node_id, 0)],
                ..NodeNetwork::default()
            },
        })
    }

    pub fn document_id_encoding(&self) -> u64 {
        self.document_id_encoding
    }

    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn network(&self) -> &NodeNetwork {
        &self.network
    }

    pub fn inputs(&self) -> Result<(f64, f64), NumericDocumentError> {
        numeric_inputs(&self.network, self.node_id)
    }

    pub fn validate_command(
        &self,
        command: &NumericAddCommand,
    ) -> Result<(), NumericDocumentError> {
        if command.document_id_encoding != self.document_id_encoding {
            return Err(NumericDocumentError::ForeignDocument);
        }
        if !command.network_path.is_empty() {
            return Err(NumericDocumentError::UnsupportedNetworkPath);
        }
        if command.target_node_id != self.node_id {
            return Err(NumericDocumentError::MissingNode);
        }
        if command.expected_revision != self.revision {
            return Err(NumericDocumentError::StaleRevision {
                expected: command.expected_revision,
                actual: self.revision,
            });
        }
        numeric_inputs(&self.network, self.node_id)?;
        if !command.value.is_finite() {
            return Err(NumericDocumentError::NonFiniteValue);
        }
        Ok(())
    }

    pub fn apply(&mut self, command: NumericAddCommand) -> Result<u64, NumericDocumentError> {
        self.validate_command(&command)?;
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or(NumericDocumentError::RevisionExhausted)?;
        let node = self
            .network
            .nodes
            .get_mut(&self.node_id)
            .ok_or(NumericDocumentError::MissingNode)?;
        let input = match command.input {
            NumericAddInput::Augend => node.input_mut(add::AugendInput),
            NumericAddInput::Addend => node.input_mut(add::AddendInput),
        }
        .ok_or(NumericDocumentError::InvalidInputType)?;
        let mut value = input
            .as_value_mut()
            .ok_or(NumericDocumentError::InvalidInputType)?;
        *value = TaggedValue::Number(command.value);
        drop(value);
        self.revision = next_revision;
        Ok(self.revision)
    }

    pub fn snapshot(&self) -> NumericAddSnapshot {
        NumericAddSnapshot {
            document_id_encoding: self.document_id_encoding,
            node_id: self.node_id,
            revision: self.revision,
            network: self.network.clone(),
        }
    }
}

pub struct NumericAddSnapshot {
    document_id_encoding: u64,
    node_id: NodeId,
    revision: u64,
    network: NodeNetwork,
}

impl NumericAddSnapshot {
    /// Retains a screened source network and caller-supplied process-local context.
    /// The revision is not inferred from a source hash or authenticated as current.
    pub fn from_network(
        document_id_encoding: u64,
        node_id: NodeId,
        revision: u64,
        network: NodeNetwork,
    ) -> Result<Self, NumericDocumentError> {
        numeric_inputs(&network, node_id)?;
        Ok(Self {
            document_id_encoding,
            node_id,
            revision,
            network,
        })
    }

    pub fn document_id_encoding(&self) -> u64 {
        self.document_id_encoding
    }

    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn network(&self) -> &NodeNetwork {
        &self.network
    }

    pub fn inputs(&self) -> Result<(f64, f64), NumericDocumentError> {
        numeric_inputs(&self.network, self.node_id)
    }

    /// Operation oracle only; imported graphs use the original evaluator below.
    pub fn evaluate(&self) -> Result<f64, NumericDocumentError> {
        let (augend, addend) = self.inputs()?;
        let node = self
            .network
            .nodes
            .get(&self.node_id)
            .ok_or(NumericDocumentError::MissingNode)?;
        let default = DocumentNode::default();
        if !node.visible
            || node.call_argument != default.call_argument
            || node.context_features != default.context_features
        {
            return Err(NumericDocumentError::InvalidNodeContract);
        }
        Ok(crate::evaluate_add(augend, addend))
    }

    pub async fn evaluate_network(&self) -> Result<f64, NumericNetworkEvaluationError> {
        contain_numeric_source_panic(self.evaluate_original_network()).await
    }

    async fn evaluate_original_network(&self) -> Result<f64, NumericNetworkEvaluationError> {
        use graphite_graph_craft::graphene_compiler::{Compiler, Executor};
        use graphite_interpreted_executor::dynamic_executor::DynamicExecutor;

        self.inputs()
            .map_err(NumericNetworkEvaluationError::Validation)?;
        let proto = Compiler {}
            .compile_single(self.network.clone())
            .map_err(NumericNetworkEvaluationError::Compilation)?;
        let executor = DynamicExecutor::new(proto)
            .await
            .map_err(NumericNetworkEvaluationError::Construction)?;
        let context: graphite_core_types::Context<'static> = None;
        let output = Executor::execute(&&executor, context)
            .await
            .map_err(NumericNetworkEvaluationError::Execution)?;
        match output {
            TaggedValue::Number(value) => Ok(value),
            output => Err(NumericNetworkEvaluationError::UnexpectedOutput(Box::new(
                output,
            ))),
        }
    }
}

async fn contain_numeric_source_panic(
    evaluation: impl std::future::Future<Output = Result<f64, NumericNetworkEvaluationError>>,
) -> Result<f64, NumericNetworkEvaluationError> {
    AssertUnwindSafe(evaluation)
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|detail| (*detail).to_owned())
                })
                .unwrap_or_else(|| "Non-string panic payload".to_owned());
            Err(NumericNetworkEvaluationError::SourcePanic(detail))
        })
}

fn numeric_inputs(
    network: &NodeNetwork,
    node_id: NodeId,
) -> Result<(f64, f64), NumericDocumentError> {
    let node = network
        .nodes
        .get(&node_id)
        .ok_or(NumericDocumentError::MissingNode)?;
    if network.nodes.len() != 1
        || network.exports != [NodeInput::node(node_id, 0)]
        || !network.scope_injections.is_empty()
        || node.inputs.len() != 2
        || node.implementation != DocumentNodeImplementation::ProtoNode(add::IDENTIFIER)
        || add::AugendInput::NODE_IDENTIFIER != add::IDENTIFIER
        || add::AddendInput::NODE_IDENTIFIER != add::IDENTIFIER
    {
        return Err(NumericDocumentError::InvalidNodeContract);
    }
    let (Some(TaggedValue::Number(augend)), Some(TaggedValue::Number(addend))) = (
        node.input_value(add::AugendInput),
        node.input_value(add::AddendInput),
    ) else {
        return Err(NumericDocumentError::InvalidInputType);
    };
    if !augend.is_finite() || !addend.is_finite() {
        return Err(NumericDocumentError::NonFiniteValue);
    }
    Ok((*augend, *addend))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> NumericAddDocument {
        NumericAddDocument::new(17, NodeId(23), 2., 3.).expect("explicit finite example")
    }

    fn command(
        document: &NumericAddDocument,
        input: NumericAddInput,
        value: f64,
    ) -> NumericAddCommand {
        NumericAddCommand {
            document_id_encoding: document.document_id_encoding(),
            network_path: Vec::new(),
            target_node_id: document.node_id(),
            expected_revision: document.revision(),
            input,
            value,
        }
    }

    fn assert_unchanged(document: &NumericAddDocument, snapshot: &NumericAddSnapshot) {
        assert_eq!(document.revision(), snapshot.revision());
        assert_eq!(document.network(), snapshot.network());
        let (augend, addend) = document.inputs().expect("numeric inputs");
        let (before_augend, before_addend) = snapshot.inputs().expect("numeric snapshot");
        assert_eq!(augend.to_bits(), before_augend.to_bits());
        assert_eq!(addend.to_bits(), before_addend.to_bits());
    }

    #[test]
    fn at_graphite_numeric_document_uses_original_add_identifier_and_port_order() {
        let document = document();
        let node = document
            .network()
            .nodes
            .get(&NodeId(23))
            .expect("only original node");
        assert_eq!(
            node.implementation,
            DocumentNodeImplementation::ProtoNode(add::IDENTIFIER)
        );
        assert_eq!(add::IDENTIFIER.as_str(), "math_nodes::AddNode");
        assert_eq!(add::AugendInput::INDEX, 0);
        assert_eq!(add::AddendInput::INDEX, 1);
        assert_eq!(
            node.input_value(add::AugendInput),
            Some(&TaggedValue::Number(2.))
        );
        assert_eq!(
            node.input_value(add::AddendInput),
            Some(&TaggedValue::Number(3.))
        );
        assert_eq!(document.network().nodes.len(), 1);
        assert_eq!(document.network().exports, [NodeInput::node(NodeId(23), 0)]);
        assert!(document.network().scope_injections.is_empty());
        assert_eq!(
            document
                .snapshot()
                .evaluate()
                .expect("original operation oracle"),
            crate::evaluate_add(2., 3.)
        );
    }

    #[test]
    fn at_graphite_numeric_document_rejects_foreign_stale_path_and_node_atomically() {
        for error in [
            NumericDocumentError::ForeignDocument,
            NumericDocumentError::UnsupportedNetworkPath,
            NumericDocumentError::MissingNode,
            NumericDocumentError::StaleRevision {
                expected: 1,
                actual: 0,
            },
        ] {
            let mut document = document();
            let before = document.snapshot();
            let mut edit = command(&document, NumericAddInput::Augend, -9.);
            match error {
                NumericDocumentError::ForeignDocument => edit.document_id_encoding = 18,
                NumericDocumentError::UnsupportedNetworkPath => {
                    edit.network_path.push(document.node_id())
                }
                NumericDocumentError::MissingNode => edit.target_node_id = NodeId(24),
                NumericDocumentError::StaleRevision { .. } => edit.expected_revision = 1,
                _ => unreachable!("finite test cases"),
            }
            assert_eq!(document.apply(edit), Err(error));
            assert_unchanged(&document, &before);
        }
    }

    #[test]
    fn at_graphite_numeric_document_rejects_nonfinite_constructor_and_edits() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                NumericAddDocument::new(0, NodeId(0), value, 0.),
                Err(NumericDocumentError::NonFiniteValue)
            ));
            assert!(matches!(
                NumericAddDocument::new(u64::MAX, NodeId(0), 0., value),
                Err(NumericDocumentError::NonFiniteValue)
            ));
            for input in [NumericAddInput::Augend, NumericAddInput::Addend] {
                let mut document = document();
                let before = document.snapshot();
                let edit = command(&document, input, value);
                assert_eq!(
                    document.apply(edit),
                    Err(NumericDocumentError::NonFiniteValue)
                );
                assert_unchanged(&document, &before);
            }
        }
        assert_eq!(
            NumericAddDocument::new(0, NodeId(0), 0., 0.)
                .expect("zero encoding preserved")
                .document_id_encoding(),
            0
        );
        assert_eq!(
            NumericAddDocument::new(u64::MAX, NodeId(u64::MAX), 0., 0.)
                .expect("full original encoding preserved")
                .document_id_encoding(),
            u64::MAX
        );
    }

    #[test]
    fn at_graphite_numeric_document_rejects_wrong_stored_type_and_contract_before_mutation() {
        let mut document = document();
        *document
            .network
            .nodes
            .get_mut(&NodeId(23))
            .expect("node")
            .input_mut(add::AddendInput)
            .expect("Addend") = NodeInput::value(TaggedValue::Bool(true), true);
        let before = document.snapshot();
        let edit = command(&document, NumericAddInput::Augend, 8.);
        assert_eq!(
            document.apply(edit),
            Err(NumericDocumentError::InvalidInputType)
        );
        assert_eq!(document.network(), before.network());
        assert_eq!(document.revision(), 0);
        assert_eq!(
            before.evaluate(),
            Err(NumericDocumentError::InvalidInputType)
        );
        document
            .network
            .nodes
            .get_mut(&NodeId(23))
            .expect("node")
            .implementation =
            DocumentNodeImplementation::ProtoNode(graphite_math_nodes::subtract::IDENTIFIER);
        let before = document.snapshot();
        let edit = command(&document, NumericAddInput::Augend, 8.);
        assert_eq!(
            document.apply(edit),
            Err(NumericDocumentError::InvalidNodeContract)
        );
        assert_eq!(document.network(), before.network());
        assert_eq!(document.revision(), 0);
    }

    #[test]
    fn at_graphite_numeric_document_snapshot_detaches_numeric_cow_values() {
        let mut document = document();
        let mut snapshot = document.snapshot();
        let snapshot_network = snapshot.network().clone();
        let edit = command(&document, NumericAddInput::Augend, 7.);
        assert_eq!(document.apply(edit), Ok(1));
        assert_eq!(snapshot.network(), &snapshot_network);
        assert_eq!(snapshot.inputs(), Ok((2., 3.)));
        assert_eq!(snapshot.evaluate().expect("old original operation"), 5.);
        let mut value = snapshot
            .network
            .nodes
            .get_mut(&NodeId(23))
            .expect("snapshot node")
            .input_mut(add::AddendInput)
            .expect("snapshot Addend")
            .as_value_mut()
            .expect("numeric COW guard");
        *value = TaggedValue::Number(-4.);
        drop(value);
        assert_eq!(snapshot.inputs(), Ok((2., -4.)));
        assert_eq!(document.inputs(), Ok((7., 3.)));
        assert_eq!(document.revision(), 1);
        assert_eq!(snapshot.revision(), 0);
        assert_eq!(snapshot.document_id_encoding(), 17);
        assert_eq!(snapshot.node_id(), NodeId(23));
    }

    #[test]
    fn at_graphite_numeric_document_preserves_signed_zero_and_second_document_isolation() {
        let mut first = NumericAddDocument::new(17, NodeId(23), 0., -0.).expect("finite");
        let second = NumericAddDocument::new(18, NodeId(23), -0., 0.).expect("separate owner");
        let second_before = second.snapshot();
        let edit = command(&first, NumericAddInput::Augend, -0.);
        assert_eq!(first.apply(edit), Ok(1));
        let values = first.inputs().expect("values");
        assert_eq!(values.0.to_bits(), (-0_f64).to_bits());
        assert_eq!(values.1.to_bits(), (-0_f64).to_bits());
        assert_eq!(
            first
                .snapshot()
                .evaluate()
                .expect("original signed-zero arithmetic")
                .to_bits(),
            crate::evaluate_add(-0., -0.).to_bits()
        );
        assert_unchanged(&second, &second_before);
        let edit = command(&second, NumericAddInput::Addend, 5.);
        let before = first.snapshot();
        assert_eq!(
            first.apply(edit),
            Err(NumericDocumentError::ForeignDocument)
        );
        assert_unchanged(&first, &before);
    }

    #[test]
    fn at_graphite_numeric_document_revision_boundary_is_atomic() {
        let mut document = document();
        document.revision = u64::MAX - 1;
        let edit = command(&document, NumericAddInput::Addend, 11.);
        assert_eq!(document.apply(edit), Ok(u64::MAX));
        let before = document.snapshot();
        let edit = command(&document, NumericAddInput::Augend, 12.);
        assert_eq!(
            document.apply(edit),
            Err(NumericDocumentError::RevisionExhausted)
        );
        assert_unchanged(&document, &before);
    }

    #[test]
    fn at_graphite_numeric_snapshot_lowers_through_original_compiler_without_owner_mutation() {
        use graphite_graph_craft::{graphene_compiler::Compiler, proto::ConstructionArgs};

        let mut document = document();
        let edit = command(&document, NumericAddInput::Addend, -0.);
        document.apply(edit).expect("addressed source edit");
        let snapshot = document.snapshot();
        let before = snapshot.network().clone();
        assert_eq!(snapshot.network().exports.len(), 1);
        let proto = Compiler {}
            .compile_single(snapshot.network().clone())
            .expect("original restricted numeric lowering");
        assert!(proto.inputs.is_empty());
        assert_eq!(proto.nodes.len(), 3);
        let add_nodes = proto
            .nodes
            .iter()
            .filter(|(_, node)| node.identifier == add::IDENTIFIER)
            .collect::<Vec<_>>();
        assert_eq!(add_nodes.len(), 1);
        let (add_id, add_node) = add_nodes.first().expect("one actual Add constructor");
        assert_eq!(proto.output, *add_id);
        assert_eq!(
            add_node.original_location.path,
            Some(vec![snapshot.node_id()])
        );
        assert_eq!(add_node.original_location.inputs_exposed, vec![true, true]);
        assert_eq!(
            add_node.call_argument,
            before
                .nodes
                .get(&snapshot.node_id())
                .expect("source node")
                .call_argument
        );
        let ConstructionArgs::Nodes(inputs) = &add_node.construction_args else {
            panic!("original Add lowering must reference its ordered value constructors");
        };
        assert_eq!(inputs.len(), 2);
        let (augend, addend) = snapshot.inputs().expect("original Number slots");
        for (index, expected) in [
            (add::AugendInput::INDEX, augend),
            (add::AddendInput::INDEX, addend),
        ] {
            let input_id = inputs.get(index).expect("source-declared port");
            let (_, input) = proto
                .nodes
                .iter()
                .find(|(node_id, _)| node_id == input_id)
                .expect("connected original value constructor");
            assert_eq!(
                input.identifier,
                graphite_core_types::ProtoNodeIdentifier::new("core_types::value::ClonedNode")
            );
            let ConstructionArgs::Value(value) = &input.construction_args else {
                panic!("original value lowering must retain Number data");
            };
            let TaggedValue::Number(actual) = &**value else {
                panic!("only the admitted Number variant may be lowered");
            };
            assert_eq!(actual.to_bits(), expected.to_bits());
            let path = input
                .original_location
                .path
                .as_ref()
                .expect("physical original value path");
            assert_eq!(path.len(), 2);
            assert_eq!(path.first(), Some(&snapshot.node_id()));
            assert_eq!(
                input.original_location.dependants,
                vec![vec![snapshot.node_id()]]
            );
        }
        assert_eq!(snapshot.document_id_encoding(), 17);
        assert_eq!(snapshot.node_id(), NodeId(23));
        assert_eq!(snapshot.revision(), 1);
        assert_eq!(snapshot.network(), &before);
        assert_unchanged(&document, &snapshot);
        let lowered_before = proto.clone();
        let edit = command(&document, NumericAddInput::Augend, 9.);
        document.apply(edit).expect("subsequent document mutation");
        assert_eq!(document.revision(), 2);
        assert_eq!(snapshot.revision(), 1);
        assert_eq!(snapshot.network(), &before);
        assert_eq!(proto, lowered_before);
    }

    #[test]
    fn at_graphite_numeric_snapshot_executes_original_network_bitwise_against_native_add() {
        use futures::executor::block_on;

        for (augend, addend) in [
            (2., 3.),
            (-9.5, 2.25),
            (-0., -0.),
            (0., -0.),
            (1., -1.),
            (f64::MAX, f64::MAX),
            (-f64::MAX, -f64::MAX),
            (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            (f64::from_bits(1), f64::from_bits(2)),
        ] {
            let document = NumericAddDocument::new(17, NodeId(23), augend, addend)
                .expect("explicit finite numeric inputs");
            let snapshot = document.snapshot();
            let before = snapshot.network().clone();
            let expected = snapshot
                .evaluate()
                .expect("existing original direct constructor");
            let actual = block_on(snapshot.evaluate_network())
                .expect("original registry construction and execution");
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "inputs {augend:?}, {addend:?}"
            );
            assert_eq!(snapshot.network(), &before);
            assert_unchanged(&document, &snapshot);
            let repeated =
                block_on(snapshot.evaluate_network()).expect("repeated original execution");
            assert_eq!(repeated.to_bits(), actual.to_bits());
        }
    }

    #[test]
    fn at_graphite_numeric_snapshot_network_execution_is_detached_and_document_isolated() {
        use futures::executor::block_on;

        let mut first = document();
        let old = first.snapshot();
        let old_network = old.network().clone();
        let second =
            NumericAddDocument::new(18, NodeId(23), 20., 30.).expect("separate document owner");
        let second_before = second.snapshot();
        let edit = command(&first, NumericAddInput::Augend, 7.);
        first.apply(edit).expect("addressed edit");
        let current = first.snapshot();
        for snapshot in [&old, &current, &second_before] {
            let expected = snapshot.evaluate().expect("direct original constructor");
            assert_eq!(
                block_on(snapshot.evaluate_network())
                    .expect("actual detached original graph execution")
                    .to_bits(),
                expected.to_bits()
            );
        }
        assert_eq!(old.network(), &old_network);
        assert_eq!(old.document_id_encoding(), 17);
        assert_eq!(old.node_id(), NodeId(23));
        assert_eq!(old.revision(), 0);
        assert_eq!(old.inputs(), Ok((2., 3.)));
        assert_eq!(current.revision(), 1);
        assert_eq!(first.inputs(), Ok((7., 3.)));
        assert_unchanged(&first, &current);
        assert_unchanged(&second, &second_before);
        assert_eq!(second.document_id_encoding(), 18);
    }

    #[test]
    fn at_graphite_numeric_snapshot_network_rejects_malformed_data_before_original_compiler() {
        use futures::executor::block_on;

        let document = document();
        let mut wrong_type = document.snapshot();
        *wrong_type
            .network
            .nodes
            .get_mut(&NodeId(23))
            .expect("private malformed snapshot node")
            .input_mut(add::AugendInput)
            .expect("source symbol") = NodeInput::value(TaggedValue::Bool(true), true);
        let before = wrong_type.network().clone();
        assert!(matches!(
            block_on(wrong_type.evaluate_network()),
            Err(NumericNetworkEvaluationError::Validation(
                NumericDocumentError::InvalidInputType
            ))
        ));
        assert_eq!(wrong_type.network(), &before);
        let mut no_export = document.snapshot();
        no_export.network.exports.clear();
        let before = no_export.network().clone();
        assert!(matches!(
            block_on(no_export.evaluate_network()),
            Err(NumericNetworkEvaluationError::Validation(
                NumericDocumentError::InvalidNodeContract
            ))
        ));
        assert_eq!(no_export.network(), &before);
        let mut nonfinite = document.snapshot();
        *nonfinite
            .network
            .nodes
            .get_mut(&NodeId(23))
            .expect("private malformed snapshot node")
            .input_mut(add::AddendInput)
            .expect("source symbol") = NodeInput::value(TaggedValue::Number(f64::INFINITY), true);
        let before = nonfinite.network().clone();
        assert!(matches!(
            block_on(nonfinite.evaluate_network()),
            Err(NumericNetworkEvaluationError::Validation(
                NumericDocumentError::NonFiniteValue
            ))
        ));
        assert_eq!(nonfinite.network(), &before);
        assert_eq!(nonfinite.revision(), 0);
        assert_unchanged(&document, &document.snapshot());
    }
    #[test]
    fn addressed_validation_does_not_mutate_or_bypass_identity_for_unchanged_values() {
        let mut document = document();
        let snapshot = document.snapshot();
        let original = document.inputs().expect("valid original inputs");
        let mut unchanged = command(&document, NumericAddInput::Augend, original.0);
        document
            .validate_command(&unchanged)
            .expect("valid unchanged command");
        assert_eq!(document.revision(), 0);
        assert_eq!(snapshot.network(), document.network());
        unchanged.document_id_encoding += 1;
        assert_eq!(
            document.validate_command(&unchanged),
            Err(NumericDocumentError::ForeignDocument)
        );
        unchanged.document_id_encoding = document.document_id_encoding();
        unchanged.network_path.push(document.node_id());
        assert_eq!(
            document.validate_command(&unchanged),
            Err(NumericDocumentError::UnsupportedNetworkPath)
        );
        unchanged.network_path.clear();
        unchanged.expected_revision = 1;
        assert_eq!(
            document.validate_command(&unchanged),
            Err(NumericDocumentError::StaleRevision {
                expected: 1,
                actual: 0
            })
        );
        unchanged.expected_revision = 0;
        unchanged.value = f64::INFINITY;
        assert_eq!(
            document.validate_command(&unchanged),
            Err(NumericDocumentError::NonFiniteValue)
        );
        assert_eq!(document.revision(), 0);
        assert_eq!(document.inputs().expect("retained inputs"), original);
        document
            .apply(command(&document, NumericAddInput::Augend, original.0))
            .expect("Core writes still advance even for identical bits");
        assert_eq!(document.revision(), 1);
        assert_eq!(snapshot.revision(), 0);
    }

    #[test]
    fn numeric_source_panic_boundary_returns_a_recoverable_error() {
        let error = futures::executor::block_on(contain_numeric_source_panic(async {
            panic!("controlled numeric source failure");
        }))
        .expect_err("panic must become an error");
        assert!(
            matches!(error, NumericNetworkEvaluationError::SourcePanic(ref detail) if detail == "controlled numeric source failure")
        );
        let document = document();
        assert_eq!(
            futures::executor::block_on(document.snapshot().evaluate_network())
                .expect("real source evaluation recovers"),
            5.
        );
    }

    #[test]
    fn numeric_import_retains_all_source_fields_and_supplied_context() {
        let mut network = document().network().clone();
        network.generated = true;
        let node = network.nodes.get_mut(&NodeId(23)).expect("source node");
        node.skip_deduplication = true;
        node.original_location.path = Some(vec![NodeId(99), NodeId(23)]);
        node.original_location.inputs_exposed = vec![false, true];
        node.original_location.auto_convert_index = Some(1);
        node.original_location.dependants = vec![vec![NodeId(51)]];
        node.original_location.inputs_source.insert(
            graphite_graph_craft::document::Source {
                node: vec![NodeId(99)],
                index: 1,
            },
            0,
        );
        let original = network.clone();
        let snapshot = NumericAddSnapshot::from_network(u64::MAX, NodeId(23), 71, network)
            .expect("screened exact network");
        assert_eq!(snapshot.network(), &original);
        assert!(snapshot.network().generated);
        assert_eq!(snapshot.document_id_encoding(), u64::MAX);
        assert_eq!(snapshot.node_id(), NodeId(23));
        assert_eq!(snapshot.revision(), 71);
        assert_eq!(snapshot.inputs(), Ok((2., 3.)));
    }

    #[test]
    fn numeric_import_preserves_hidden_passthrough_semantics() {
        let mut network = document().network().clone();
        network
            .nodes
            .get_mut(&NodeId(23))
            .expect("source node")
            .visible = false;
        let original = network.clone();
        let snapshot = NumericAddSnapshot::from_network(17, NodeId(23), 4, network)
            .expect("original hidden Add network");
        assert_eq!(
            snapshot.evaluate(),
            Err(NumericDocumentError::InvalidNodeContract)
        );
        assert_eq!(
            futures::executor::block_on(snapshot.evaluate_network())
                .expect("original hidden passthrough"),
            2.
        );
        assert_eq!(snapshot.network(), &original);
        assert!(
            !snapshot
                .network()
                .nodes
                .get(&NodeId(23))
                .expect("retained source")
                .visible
        );
        assert_eq!(snapshot.revision(), 4);
    }

    #[test]
    fn numeric_import_does_not_normalize_nondefault_call_or_context() {
        for change_call in [true, false] {
            let mut network = document().network().clone();
            let node = network.nodes.get_mut(&NodeId(23)).expect("source node");
            if change_call {
                node.call_argument = graphite_core_types::Type::default();
            } else {
                node.context_features.extract = graphite_core_types::ContextFeatures::FOOTPRINT;
            }
            let original = network.clone();
            let snapshot = NumericAddSnapshot::from_network(17, NodeId(23), 8, network)
                .expect("source contract with nondefault execution fields");
            assert_eq!(snapshot.network(), &original);
            assert_eq!(
                snapshot.evaluate(),
                Err(NumericDocumentError::InvalidNodeContract)
            );
            assert_eq!(snapshot.network(), &original);
        }
    }

    #[test]
    fn numeric_import_rejects_invalid_networks_without_source_repair() {
        for case in 0..5 {
            let mut network = document().network().clone();
            let node_id = if case == 0 { NodeId(999) } else { NodeId(23) };
            let expected = match case {
                0 => NumericDocumentError::MissingNode,
                1 => {
                    network.exports.clear();
                    NumericDocumentError::InvalidNodeContract
                }
                2 => {
                    network.nodes.get_mut(&node_id).expect("source node").inputs[0] =
                        NodeInput::value(TaggedValue::Bool(true), true);
                    NumericDocumentError::InvalidInputType
                }
                3 => {
                    network.nodes.get_mut(&node_id).expect("source node").inputs[1] =
                        NodeInput::value(TaggedValue::Number(f64::NAN), true);
                    NumericDocumentError::NonFiniteValue
                }
                4 => {
                    network.scope_injections.insert(
                        "source scope".to_owned(),
                        (node_id, graphite_core_types::Type::default()),
                    );
                    NumericDocumentError::InvalidNodeContract
                }
                _ => unreachable!(),
            };
            assert!(
                matches!(NumericAddSnapshot::from_network(17, node_id, 8, network),
                Err(ref error) if *error == expected)
            );
        }
    }
}
