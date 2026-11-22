use std::{collections::BTreeMap, fmt, sync::OnceLock};

use graphite_core_types::{
    Color, NodeIOTypes, ProtoNodeIdentifier, list,
    registry::{NODE_METADATA, NODE_REGISTRY, NodeMetadata},
};
use graphite_editor::messages::portfolio::document::{
    node_graph::{
        document_node_definitions::{
            DocumentNodeDefinition, collect_node_types, resolve_network_node_type,
        },
        utility_types::{FrontendNodeType, InputTypeConstraint},
    },
    utility_types::network_interface::{
        NodeNetworkTemplate, NodeTemplateImplementation, NodeTypePersistentMetadata,
    },
};
use graphite_graph_craft::document::{
    NodeId, NodeInput,
    value::{DVec2, TaggedValue},
};
use graphite_std::{Artboard, Graphic};

#[derive(Clone, Debug)]
pub struct RegisteredNode {
    pub metadata: NodeMetadata,
    pub signatures: Vec<NodeIOTypes>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryError {
    MetadataTablePoisoned,
    ConstructorTablePoisoned,
    MissingSourceNode,
    SourceCatalogUnavailable,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataTablePoisoned => {
                formatter.write_str("The source node metadata table is poisoned")
            }
            Self::ConstructorTablePoisoned => {
                formatter.write_str("The source node constructor table is poisoned")
            }
            Self::MissingSourceNode => {
                formatter.write_str("The node is missing from the source registry")
            }
            Self::SourceCatalogUnavailable => {
                formatter.write_str("The pinned Graphite node catalog could not be loaded")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

pub fn registered_node(identifier: &str) -> Result<RegisteredNode, RegistryError> {
    let identifier = ProtoNodeIdentifier::with_owned_string(identifier.to_owned());
    let metadata = {
        let metadata = NODE_METADATA
            .lock()
            .map_err(|_| RegistryError::MetadataTablePoisoned)?;
        metadata.get(&identifier).cloned()
    };
    let signatures = {
        let registry = NODE_REGISTRY
            .lock()
            .map_err(|_| RegistryError::ConstructorTablePoisoned)?;
        registry.get(&identifier).map(|constructors| {
            constructors
                .iter()
                .map(|(_, signature)| signature.clone())
                .collect()
        })
    };
    Ok(RegisteredNode {
        metadata: metadata.ok_or(RegistryError::MissingSourceNode)?,
        signatures: signatures.ok_or(RegistryError::MissingSourceNode)?,
    })
}

pub const CUSTOM_NODE_NAME: &str = "Custom Node";
pub const CUSTOM_NODE_IDENTIFIER: &str = "NETWORK:Custom Node";
pub(crate) const MERGE_NODE_NAME: &str = "Merge";
pub(crate) const MERGE_NODE_IDENTIFIER: &str = "NETWORK:Merge";
pub(crate) const ARTBOARD_NODE_NAME: &str = "Artboard";
pub(crate) const ARTBOARD_NODE_IDENTIFIER: &str = "NETWORK:Artboard";

pub(crate) fn custom_node_definition() -> Option<&'static DocumentNodeDefinition> {
    let definition = resolve_network_node_type(CUSTOM_NODE_NAME)?;
    let template = &definition.node_template;
    let NodeTemplateImplementation::Network(network) = &template.implementation else {
        return None;
    };
    let expected = NodeNetworkTemplate {
        reference: Some(definition.identifier.into()),
        ..Default::default()
    };
    (template.inputs.is_empty() && *network == expected).then_some(definition)
}

pub(crate) fn merge_node_definition() -> Option<&'static DocumentNodeDefinition> {
    let definition = resolve_network_node_type(MERGE_NODE_NAME)?;
    let template = &definition.node_template;
    let NodeTemplateImplementation::Network(network) = &template.implementation else {
        return None;
    };
    let input = NodeInput::type_default(list!(Graphic), true);
    (matches!(
        template.node_type_metadata,
        NodeTypePersistentMetadata::Layer(_)
    ) && template.inputs.as_slice() == [input.clone(), input]
        && template.output_names.as_slice() == ["Out"]
        && network.reference.as_deref() == Some(MERGE_NODE_NAME)
        && network.scope_injections.is_empty()
        && network.nodes.len() == 6
        && network.exports.as_slice() == [NodeInput::node(NodeId(5), 0)])
    .then_some(definition)
}

pub(crate) fn artboard_node_definition() -> Option<&'static DocumentNodeDefinition> {
    let definition = resolve_network_node_type(ARTBOARD_NODE_NAME)?;
    let template = &definition.node_template;
    let NodeTemplateImplementation::Network(network) = &template.implementation else {
        return None;
    };
    (matches!(
        template.node_type_metadata,
        NodeTypePersistentMetadata::Layer(_)
    ) && template.inputs.as_slice()
        == [
            NodeInput::type_default(list!(Artboard), true),
            NodeInput::type_default(list!(Graphic), true),
            NodeInput::value(TaggedValue::DVec2(DVec2::ZERO), false),
            NodeInput::value(TaggedValue::DVec2(DVec2::new(1920., 1080.)), false),
            NodeInput::value(TaggedValue::Color(Color::WHITE), false),
            NodeInput::value(TaggedValue::Bool(true), false),
        ]
        && template.output_names.as_slice() == ["Out"]
        && network.reference.as_deref() == Some(ARTBOARD_NODE_NAME)
        && network.scope_injections.is_empty()
        && network.nodes.len() == 5
        && network.exports.as_slice() == [NodeInput::node(NodeId(4), 0)])
    .then_some(definition)
}

pub struct GraphiteCatalogCategory {
    pub name: &'static str,
    pub nodes: Vec<&'static FrontendNodeType>,
}

pub struct GraphiteNodeCatalog {
    entries: &'static [FrontendNodeType],
}

impl GraphiteNodeCatalog {
    pub fn load() -> Result<Self, RegistryError> {
        static SOURCE: OnceLock<Vec<FrontendNodeType>> = OnceLock::new();
        let entries = std::panic::catch_unwind(|| SOURCE.get_or_init(collect_node_types))
            .map_err(|_| RegistryError::SourceCatalogUnavailable)?;
        Ok(Self { entries })
    }

    pub fn search(&self, query: &str) -> Vec<GraphiteCatalogCategory> {
        let lower = query.to_lowercase();
        let mut type_term = None;
        let mut terms = vec![lower.as_str()];
        if let Some(type_search) = lower.strip_prefix("type:") {
            let mut words = type_search.split_whitespace();
            type_term = words.next();
            terms = words.collect();
        }
        let mut categories = BTreeMap::<&str, Vec<&FrontendNodeType>>::new();
        for node in self.entries {
            if type_term.is_some_and(|term| {
                !node.input_types.iter().any(|constraint| match constraint {
                    InputTypeConstraint::All => true,
                    InputTypeConstraint::Limited(types) => types
                        .iter()
                        .any(|input_type| input_type.to_lowercase().contains(term)),
                })
            }) {
                continue;
            }
            let name = node.name.to_lowercase();
            let category = node.category.to_lowercase();
            if !terms.iter().all(|term| {
                name.contains(term)
                    || category.contains(term)
                    || (node.name == MERGE_NODE_NAME && "layer".contains(term))
            }) {
                continue;
            }
            categories.entry(&node.category).or_default().push(node);
        }
        let mut groups = categories
            .into_iter()
            .map(|(name, nodes)| GraphiteCatalogCategory { name, nodes })
            .collect::<Vec<_>>();
        groups
            .sort_by_cached_key(|group| (catalog_priority(group.name), group.name.to_lowercase()));
        groups
    }
}

fn catalog_priority(category: &str) -> usize {
    const FIRST_CATEGORIES: [&str; 5] = ["UNCATEGORIZED", "General", "Value", "Math", "Style"];
    if category.starts_with("Debug") {
        return FIRST_CATEGORIES.len() + 1;
    }
    FIRST_CATEGORIES
        .iter()
        .position(|prefix| category.starts_with(prefix))
        .unwrap_or(FIRST_CATEGORIES.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use graphite_core_types::{
        Color, Context, NodeParameter, Type, concrete, fn_type_fut, item, list,
        list::{Item, List},
        registry::{RegistryValueSource, RegistryWidgetOverride},
    };
    use graphite_glam::DVec2;
    use graphite_math_nodes::{
        absolute_value, ceiling, divide, floor, reciprocal, root, round, sign, subtract,
    };
    use graphite_raster_nodes::adjustments::{invert, make_opaque};
    use graphite_raster_types::{CPU, Raster};
    use graphite_vector_types::Gradient;

    #[test]
    fn at_gra_node_280_1_joins_the_pinned_empty_custom_network() {
        let catalog = GraphiteNodeCatalog::load().expect("pinned catalog");
        let groups = catalog.search("Custom Node");
        let node = groups
            .iter()
            .flat_map(|group| &group.nodes)
            .find(|node| node.identifier == CUSTOM_NODE_IDENTIFIER)
            .expect("exact serialized identity");
        assert_eq!(node.category, "General");
        assert!(node.input_types.is_empty());
        let definition = custom_node_definition().expect("pinned Custom Node definition");
        let template = definition.default_node_template();
        assert!(template.inputs.is_empty());
        assert!(template.input_metadata.is_empty());
        assert!(template.output_names.is_empty());
        let NodeTemplateImplementation::Network(network) = template.implementation else {
            panic!("pinned Custom Node network");
        };
        assert!(network.nodes.is_empty());
        assert!(network.exports.is_empty());
        assert_eq!(network.reference.as_deref(), Some("Custom Node"));
    }

    #[test]
    fn at_gra_node_281_1_resolves_the_pinned_merge_template_and_ports() {
        use graphite_graph_craft::document::{DocumentNodeMetadata, value::TaggedValue};
        use graphite_std::{ATTR_EDITOR_LAYER_PATH, graphic, memo};

        let catalog = GraphiteNodeCatalog::load().expect("pinned catalog");
        let groups = catalog.search("layer");
        let entry = groups
            .iter()
            .flat_map(|group| &group.nodes)
            .find(|node| node.identifier == MERGE_NODE_IDENTIFIER)
            .expect("canonical Merge entry");
        assert_eq!(entry.category, "General");
        assert_eq!(entry.input_types.len(), 2);
        assert_eq!(entry.name, MERGE_NODE_NAME);
        let definition = merge_node_definition().expect("pinned Merge definition");
        let template = definition.default_node_template();
        assert_eq!(
            template.inputs,
            vec![NodeInput::type_default(list!(Graphic), true); 2]
        );
        assert_eq!(template.output_names, ["Out"]);
        assert_eq!(
            template
                .input_metadata
                .iter()
                .map(|port| port.persistent_metadata.input_name.as_str())
                .collect::<Vec<_>>(),
            ["Base", "Content"]
        );
        assert_eq!(
            template.node_type_metadata,
            NodeTypePersistentMetadata::layer(graphite_glam::IVec2::ZERO)
        );
        let NodeTemplateImplementation::Network(network) = template.implementation else {
            panic!("exact Merge network")
        };
        assert_eq!(network.reference.as_deref(), Some(MERGE_NODE_NAME));
        assert_eq!(network.exports, [NodeInput::node(NodeId(5), 0)]);
        assert_eq!(network.nodes.len(), 6);
        for (index, identifier, inputs) in [
            (
                0,
                graphic::as_graphic::IDENTIFIER,
                vec![NodeInput::import(graphite_graph_craft::generic!(T), 0)],
            ),
            (
                1,
                graphic::into_group::IDENTIFIER,
                vec![NodeInput::import(graphite_graph_craft::generic!(T), 1)],
            ),
            (
                2,
                graphic::path_of_subgraph::IDENTIFIER,
                vec![NodeInput::Reflection(
                    DocumentNodeMetadata::DocumentNodePath,
                )],
            ),
            (
                3,
                graphic::write_attribute::IDENTIFIER,
                vec![
                    NodeInput::node(NodeId(1), 0),
                    NodeInput::value(TaggedValue::String(ATTR_EDITOR_LAYER_PATH.into()), false),
                    NodeInput::node(NodeId(2), 0),
                ],
            ),
            (
                4,
                memo::monitor::IDENTIFIER,
                vec![NodeInput::node(NodeId(3), 0)],
            ),
            (
                5,
                graphic::extend::IDENTIFIER,
                vec![NodeInput::node(NodeId(0), 0), NodeInput::node(NodeId(4), 0)],
            ),
        ] {
            let child = network
                .nodes
                .get(&NodeId(index))
                .expect("original internal node");
            assert_eq!(
                child.implementation,
                NodeTemplateImplementation::ProtoNode(identifier)
            );
            assert_eq!(child.inputs, inputs);
            assert_eq!(child.skip_deduplication, index == 4);
        }
    }

    #[test]
    fn at_gra_node_282_1_resolves_the_pinned_artboard_ports_and_widgets() {
        use graphite_graph_craft::document::DocumentNodeMetadata;

        let catalog = GraphiteNodeCatalog::load().expect("pinned catalog");
        let groups = catalog.search("Artboard");
        let entry = groups
            .iter()
            .flat_map(|group| &group.nodes)
            .find(|node| node.identifier == ARTBOARD_NODE_IDENTIFIER)
            .expect("canonical Artboard entry");
        assert_eq!(entry.name, ARTBOARD_NODE_NAME);
        assert_eq!(entry.category, "General");
        let template = artboard_node_definition()
            .expect("pinned Artboard")
            .default_node_template();
        let [base, content, location, dimensions, background, clip] =
            template.input_metadata.as_slice()
        else {
            panic!("six ordered Artboard properties")
        };
        for (port, name, widget) in [
            (base, "Base", None),
            (content, "Content", Some("hidden")),
            (location, "Location", Some("vec2")),
            (dimensions, "Dimensions", Some("vec2")),
            (background, "Background", Some("artboard_background")),
            (clip, "Clip", None),
        ] {
            let metadata = &port.persistent_metadata;
            assert_eq!(metadata.input_name, name);
            assert_eq!(metadata.widget_override.as_deref(), widget);
        }
        for (port, x, y) in [(location, "X", "Y"), (dimensions, "W", "H")] {
            let data = &port.persistent_metadata.input_data;
            for (key, value) in [
                ("x", serde_json::json!(x)),
                ("y", serde_json::json!(y)),
                ("unit", serde_json::json!(" px")),
                ("is_integer", serde_json::json!(true)),
            ] {
                assert_eq!(data.get(key), Some(&value));
            }
        }
        let NodeTemplateImplementation::Network(network) = template.implementation else {
            panic!("exact Artboard network")
        };
        for (index, identifier) in [
            (0, "graphic_nodes::artboard::CreateArtboardNode"),
            (1, "graphic_nodes::graphic::PathOfSubgraphNode"),
            (2, "graphic_nodes::graphic::WriteAttributeNode"),
            (3, "graphene_core::memo::MonitorNode"),
            (4, "graphic_nodes::graphic::ExtendNode"),
        ] {
            let child = network
                .nodes
                .get(&NodeId(index))
                .expect("original internal node");
            assert!(
                matches!(&child.implementation, NodeTemplateImplementation::ProtoNode(actual) if actual.as_str() == identifier)
            );
            assert_eq!(child.skip_deduplication, index == 3);
        }
        let reflection = network.nodes.get(&NodeId(1)).expect("layer provenance");
        assert_eq!(
            reflection.inputs,
            [NodeInput::Reflection(
                DocumentNodeMetadata::DocumentNodePath
            )]
        );
        let create = network
            .nodes
            .get(&NodeId(0))
            .expect("source Create Artboard");
        assert_eq!(create.inputs.len(), 5);
        assert!(create.inputs.iter().zip(1..=5).all(|(input, index)|
            matches!(input, NodeInput::Import { import_index, .. } if *import_index == index)));
    }

    #[test]
    fn at_gra_graph_001_1_catalog_keeps_all_source_visible_entries_and_categories() {
        let catalog = GraphiteNodeCatalog::load().expect("pinned node catalog");
        let actual = catalog.search("");
        let original = collect_node_types();
        assert!(!actual.is_empty());
        assert!(actual.iter().all(|category| !category.name.is_empty()));
        assert!(actual.windows(2).all(|pair| {
            let [first, second] = pair else {
                return false;
            };
            (catalog_priority(first.name), first.name)
                <= (catalog_priority(second.name), second.name)
        }));
        assert_eq!(
            actual
                .iter()
                .map(|category| category.nodes.len())
                .sum::<usize>(),
            original.len()
        );
        for group in actual {
            assert_eq!(
                group.nodes,
                original
                    .iter()
                    .filter(|node| node.category == group.name)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn at_gra_graph_002_003_004_1_catalog_search_combines_source_text_types_and_layer_alias() {
        let catalog = GraphiteNodeCatalog::load().expect("pinned node catalog");
        let arithmetic = catalog.search("mAtH: ArItHmEtIc");
        assert!(arithmetic.iter().any(|group| {
            group.name == "Math: Arithmetic" && group.nodes.iter().any(|node| node.name == "Add")
        }));
        let limited = catalog.search("TYPE: Number math subtract");
        assert!(limited.iter().flat_map(|group| &group.nodes).any(|node| {
            node.name == "Subtract" && node.input_types.iter().any(|constraint| {
                            matches!(constraint, InputTypeConstraint::Limited(types) if types.contains("Number"))
            })
        }));
        let unconstrained = catalog.search("type: application-defined-unknown-type");
        assert!(!unconstrained.is_empty());
        assert!(
            unconstrained
                .iter()
                .flat_map(|group| &group.nodes)
                .all(|node| { node.input_types.contains(&InputTypeConstraint::All) })
        );
        let merge = catalog
            .entries
            .iter()
            .find(|node| node.name == "Merge")
            .expect("source Merge");
        let alias = catalog.search("lAyEr");
        assert!(
            alias
                .iter()
                .flat_map(|group| &group.nodes)
                .any(|node| node.identifier == merge.identifier)
        );
        assert!(
            catalog
                .search("zqx nowhere in the source catalog")
                .is_empty()
        );
    }

    fn item_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(Item<T>))),
            vec![fn_type_fut!(Context<'static>, Item<T>)],
        )
    }

    fn list_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(List<T>))),
            vec![fn_type_fut!(Context<'static>, List<T>)],
        )
    }

    fn binary_signature(left: Type, right: Type, output: Type) -> NodeIOTypes {
        ranked_signature([left, right], output)
    }

    fn ranked_signature(inputs: impl IntoIterator<Item = Type>, output: Type) -> NodeIOTypes {
        let input = |value| {
            Type::Fn(
                Box::new(concrete!(Context<'static>)),
                Box::new(Type::Future(Box::new(value))),
            )
        };
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(output)),
            inputs.into_iter().map(input).collect(),
        )
    }

    #[test]
    fn at_gra_node_079_080_083_1_resolves_power_log_and_modulo_profiles() {
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_proto_node_type;
        use graphite_graph_craft::document::{NodeInput, value::TaggedValue};
        use graphite_math_nodes::{exponent, logarithm, modulo};

        fn signatures<Left: 'static, Right: 'static, Output: 'static>(
            always_positive: bool,
        ) -> [NodeIOTypes; 2] {
            [
                ranked_signature(
                    [item!(Left), item!(Right)]
                        .into_iter()
                        .chain(always_positive.then(|| item!(bool))),
                    concrete!(Item<Output>),
                ),
                ranked_signature(
                    [list!(Left), list!(Right)]
                        .into_iter()
                        .chain(always_positive.then(|| list!(bool))),
                    concrete!(List<Output>),
                ),
            ]
        }

        for (identifier, label, names) in [
            (exponent::IDENTIFIER, "Exponent", ["Base", "Power"]),
            (logarithm::IDENTIFIER, "Logarithm", ["Value", "Base"]),
            (modulo::IDENTIFIER, "Modulo", ["Numerator", "Modulus"]),
        ] {
            assert_eq!(identifier.as_str(), format!("math_nodes::{label}Node"));
            let registered =
                registered_node(identifier.as_str()).expect("pinned arithmetic operation");
            assert_eq!(registered.metadata.display_name, label);
            assert_eq!(registered.metadata.category, "Math: Arithmetic");
            assert!(registered.metadata.context_features.is_empty());
            assert_eq!(registered.metadata.properties, None);
            let has_flag = identifier == modulo::IDENTIFIER;
            let fields = &registered.metadata.fields;
            assert_eq!(fields.len(), if has_flag { 3 } else { 2 });
            let [left, right, ..] = fields.as_slice() else {
                panic!("ordered arithmetic inputs")
            };
            assert!(
                fields
                    .iter()
                    .all(|field| matches!(field.widget_override, RegistryWidgetOverride::None))
            );
            let mut expected_inputs = vec![
                NodeInput::value(TaggedValue::Numbers(Vec::new()), true),
                NodeInput::value(TaggedValue::Number(2.), false),
            ];
            let [left_name, right_name] = names;
            assert_eq!(left.name, left_name);
            assert_eq!(left.default_type, Some(list!(f64)));
            assert!(matches!(left.value_source, RegistryValueSource::None));
            assert_eq!(right.name, right_name);
            assert_eq!(right.default_type, Some(concrete!(f64)));
            assert!(matches!(
                right.value_source,
                RegistryValueSource::Default("2.")
            ));
            if has_flag {
                let flag = fields
                    .get(modulo::AlwaysPositiveInput::INDEX)
                    .expect("Always Positive port");
                assert_eq!(flag.name, "Always Positive");
                assert_eq!(flag.default_type, Some(concrete!(bool)));
                assert!(matches!(
                    flag.value_source,
                    RegistryValueSource::Default("true")
                ));
                expected_inputs.push(NodeInput::value(TaggedValue::Bool(true), false));
            }
            let definition =
                resolve_proto_node_type(identifier.clone()).expect("exact pinned node template");
            let template = definition.default_node_template();
            assert_eq!(
                template.implementation,
                NodeTemplateImplementation::ProtoNode(identifier.clone())
            );
            assert_eq!(template.inputs, expected_inputs);

            let mut expected = HashSet::new();
            expected.extend(signatures::<f64, f64, f64>(has_flag));
            expected.extend(signatures::<f64, DVec2, DVec2>(has_flag));
            expected.extend(signatures::<DVec2, f64, DVec2>(has_flag));
            expected.extend(signatures::<DVec2, DVec2, DVec2>(has_flag));
            if identifier != logarithm::IDENTIFIER {
                expected.extend(signatures::<i64, i64, i64>(has_flag));
            }
            assert_eq!(registered.signatures.len(), expected.len());
            assert_eq!(
                registered.signatures.into_iter().collect::<HashSet<_>>(),
                expected
            );
            assert!(matches!(
                registered_node(label),
                Err(RegistryError::MissingSourceNode)
            ));
        }
    }

    #[test]
    fn at_gra_node_125_to_130_1_resolves_trigonometric_ports_and_overloads() {
        use graphite_editor::messages::portfolio::document::node_graph::document_node_definitions::resolve_proto_node_type;
        use graphite_graph_craft::document::{NodeInput, value::TaggedValue};
        use graphite_math_nodes::{
            cosine, cosine_inverse, sine, sine_inverse, tangent, tangent_inverse,
        };

        for (identifier, canonical_name, label, first_port, bounded) in [
            (
                cosine::IDENTIFIER,
                "math_nodes::CosineNode",
                "Cosine",
                "Theta",
                false,
            ),
            (
                cosine_inverse::IDENTIFIER,
                "math_nodes::CosineInverseNode",
                "Cosine Inverse",
                "Value",
                true,
            ),
            (
                sine::IDENTIFIER,
                "math_nodes::SineNode",
                "Sine",
                "Theta",
                false,
            ),
            (
                sine_inverse::IDENTIFIER,
                "math_nodes::SineInverseNode",
                "Sine Inverse",
                "Value",
                true,
            ),
            (
                tangent::IDENTIFIER,
                "math_nodes::TangentNode",
                "Tangent",
                "Theta",
                false,
            ),
            (
                tangent_inverse::IDENTIFIER,
                "math_nodes::TangentInverseNode",
                "Tangent Inverse",
                "Value",
                false,
            ),
        ] {
            assert_eq!(identifier.as_str(), canonical_name);
            let registered =
                registered_node(identifier.as_str()).expect("pinned trigonometric node");
            assert_eq!(registered.metadata.display_name, label);
            assert_eq!(registered.metadata.category, "Math: Trig");
            assert_eq!(registered.metadata.properties, None);
            assert!(registered.metadata.context_features.is_empty());
            let [value, radians] = registered.metadata.fields.as_slice() else {
                panic!("two ordered input ports for {label}");
            };
            assert_eq!(value.name, first_port);
            assert_eq!(
                value.default_type,
                Some(if bounded { item!(f64) } else { list!(f64) })
            );
            assert_eq!(value.number_hard_min, bounded.then_some(-1.));
            assert_eq!(value.number_hard_max, bounded.then_some(1.));
            assert_eq!(radians.name, "Radians");
            assert_eq!(radians.default_type, Some(item!(bool)));
            for field in [value, radians] {
                assert!(matches!(field.value_source, RegistryValueSource::None));
                assert!(matches!(
                    field.widget_override,
                    RegistryWidgetOverride::None
                ));
            }
            let definition =
                resolve_proto_node_type(identifier.clone()).expect("pinned definition");
            let template = definition.default_node_template();
            assert_eq!(
                template.implementation,
                NodeTemplateImplementation::ProtoNode(identifier.clone())
            );
            assert_eq!(
                template.inputs,
                vec![
                    NodeInput::value(
                        if bounded {
                            TaggedValue::Number(0.)
                        } else {
                            TaggedValue::Numbers(Vec::new())
                        },
                        true
                    ),
                    NodeInput::value(TaggedValue::Bool(false), false),
                ]
            );

            let mut signatures = HashSet::from([
                binary_signature(item!(f64), item!(bool), concrete!(Item<f64>)),
                binary_signature(list!(f64), list!(bool), concrete!(List<f64>)),
            ]);
            if !bounded {
                let (item_output, list_output) = if identifier == tangent_inverse::IDENTIFIER {
                    (concrete!(Item<f64>), concrete!(List<f64>))
                } else {
                    (concrete!(Item<DVec2>), concrete!(List<DVec2>))
                };
                signatures.extend([
                    binary_signature(item!(DVec2), item!(bool), item_output),
                    binary_signature(list!(DVec2), list!(bool), list_output),
                ]);
            }
            assert_eq!(registered.signatures.len(), signatures.len(), "{label}");
            assert_eq!(
                registered.signatures.into_iter().collect::<HashSet<_>>(),
                signatures,
                "{label}"
            );
            assert!(matches!(
                registered_node(label),
                Err(RegistryError::MissingSourceNode)
            ));
        }
    }

    #[test]
    fn at_gra_node_101_103_113_1_rounding_keeps_exact_source_ports_and_signatures() {
        for (id, input_owner, input_index, name, expected_id) in [
            (
                ceiling::IDENTIFIER,
                ceiling::ValueInput::NODE_IDENTIFIER,
                ceiling::ValueInput::INDEX,
                "Ceiling",
                "math_nodes::CeilingNode",
            ),
            (
                floor::IDENTIFIER,
                floor::ValueInput::NODE_IDENTIFIER,
                floor::ValueInput::INDEX,
                "Floor",
                "math_nodes::FloorNode",
            ),
            (
                round::IDENTIFIER,
                round::ValueInput::NODE_IDENTIFIER,
                round::ValueInput::INDEX,
                "Round",
                "math_nodes::RoundNode",
            ),
        ] {
            assert_eq!(id.as_str(), expected_id);
            assert_eq!(input_owner, id);
            assert_eq!(input_index, 0);
            assert_unary_numeric_schema(expected_id, name, []);
        }
    }

    fn assert_unary_numeric_schema(
        identifier: &str,
        name: &str,
        extra_signatures: impl IntoIterator<Item = NodeIOTypes>,
    ) {
        let registered = registered_node(identifier).expect("pinned numeric node");
        assert_eq!(registered.metadata.display_name, name);
        assert_eq!(registered.metadata.category, "Math: Numeric");
        assert_eq!(registered.metadata.properties, None);
        assert!(registered.metadata.context_features.is_empty());
        let [input] = registered.metadata.fields.as_slice() else {
            panic!("expected the original Value port");
        };
        assert_eq!(input.name, "Value");
        assert_eq!(input.default_type, Some(list!(f64)));
        assert!(matches!(input.value_source, RegistryValueSource::None));
        assert!(matches!(
            input.widget_override,
            RegistryWidgetOverride::None
        ));
        assert!(!input.hidden && !input.exposed);
        let mut expected = HashSet::from([
            item_signature::<f64>(),
            item_signature::<DVec2>(),
            list_signature::<f64>(),
            list_signature::<DVec2>(),
        ]);
        expected.extend(extra_signatures);
        assert_eq!(registered.signatures.len(), expected.len());
        assert_eq!(
            registered.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
    }

    #[test]
    fn at_gra_node_099_1_absolute_value_registers_exactly_six_source_signatures() {
        let identifier = absolute_value::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::AbsoluteValueNode");
        assert_eq!(absolute_value::ValueInput::NODE_IDENTIFIER, identifier);
        assert_eq!(absolute_value::ValueInput::INDEX, 0);
        assert_unary_numeric_schema(
            identifier.as_str(),
            "Absolute Value",
            [item_signature::<i64>(), list_signature::<i64>()],
        );
    }

    #[test]
    fn at_gra_node_114_1_sign_registers_exactly_four_source_signatures() {
        let identifier = sign::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::SignNode");
        assert_eq!(sign::ValueInput::NODE_IDENTIFIER, identifier);
        assert_eq!(sign::ValueInput::INDEX, 0);
        assert_unary_numeric_schema(identifier.as_str(), "Sign", []);
    }

    #[test]
    fn at_gra_node_078_1_divide_exposes_exact_source_ports_and_ten_signatures() {
        let identifier = divide::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::DivideNode");
        assert_eq!(divide::NumeratorInput::NODE_IDENTIFIER, identifier);
        assert_eq!(divide::NumeratorInput::INDEX, 0);
        assert_eq!(divide::DenominatorInput::NODE_IDENTIFIER, identifier);
        assert_eq!(divide::DenominatorInput::INDEX, 1);
        let registered = registered_node(identifier.as_str()).expect("native Divide entry");
        assert_eq!(registered.metadata.display_name, "Divide");
        assert_eq!(registered.metadata.category, "Math: Arithmetic");
        assert_eq!(registered.metadata.properties, None);
        assert!(registered.metadata.context_features.is_empty());
        let [numerator, denominator] = registered.metadata.fields.as_slice() else {
            panic!("expected the two source Divide ports");
        };
        assert_eq!(numerator.name, "Numerator");
        assert_eq!(numerator.default_type, Some(list!(f64)));
        assert!(matches!(numerator.value_source, RegistryValueSource::None));
        assert_eq!(denominator.name, "Denominator");
        assert_eq!(denominator.default_type, Some(concrete!(f64)));
        assert!(matches!(
            denominator.value_source,
            RegistryValueSource::Default("1.")
        ));
        assert!(matches!(
            numerator.widget_override,
            RegistryWidgetOverride::None
        ));
        assert!(matches!(
            denominator.widget_override,
            RegistryWidgetOverride::None
        ));
        let expected = HashSet::from([
            binary_signature(item!(f64), item!(f64), concrete!(Item<f64>)),
            binary_signature(item!(i64), item!(i64), concrete!(Item<i64>)),
            binary_signature(item!(DVec2), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(item!(DVec2), item!(f64), concrete!(Item<DVec2>)),
            binary_signature(item!(f64), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(list!(f64), list!(f64), concrete!(List<f64>)),
            binary_signature(list!(i64), list!(i64), concrete!(List<i64>)),
            binary_signature(list!(DVec2), list!(DVec2), concrete!(List<DVec2>)),
            binary_signature(list!(DVec2), list!(f64), concrete!(List<DVec2>)),
            binary_signature(list!(f64), list!(DVec2), concrete!(List<DVec2>)),
        ]);
        assert_eq!(registered.signatures.len(), expected.len());
        assert_eq!(
            registered.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
    }

    #[test]
    fn at_gra_node_085_1_reciprocal_has_the_exact_source_port_and_four_signatures() {
        let identifier = reciprocal::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::ReciprocalNode");
        assert_eq!(reciprocal::ValueInput::NODE_IDENTIFIER, identifier);
        assert_eq!(reciprocal::ValueInput::INDEX, 0);
        let registered = registered_node(identifier.as_str()).expect("native Reciprocal entry");
        assert_eq!(registered.metadata.display_name, "Reciprocal");
        assert_eq!(registered.metadata.category, "Math: Arithmetic");
        assert_eq!(registered.metadata.properties, None);
        assert!(registered.metadata.context_features.is_empty());
        assert_eq!(registered.metadata.fields.len(), 1);
        let input = registered.metadata.fields.first().expect("the Value port");
        assert_eq!(input.name, "Value");
        assert_eq!(input.default_type, Some(list!(f64)));
        assert!(matches!(input.value_source, RegistryValueSource::None));
        assert!(matches!(
            input.widget_override,
            RegistryWidgetOverride::None
        ));
        let expected = HashSet::from([
            item_signature::<f64>(),
            item_signature::<DVec2>(),
            list_signature::<f64>(),
            list_signature::<DVec2>(),
        ]);
        assert_eq!(registered.signatures.len(), expected.len());
        assert_eq!(
            registered.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
    }

    #[test]
    fn at_gra_node_086_1_root_has_exact_native_defaults_and_signatures() {
        let identifier = root::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::RootNode");
        assert_eq!(root::RadicandInput::NODE_IDENTIFIER, identifier);
        assert_eq!(root::RadicandInput::INDEX, 0);
        assert_eq!(root::DegreeInput::NODE_IDENTIFIER, identifier);
        assert_eq!(root::DegreeInput::INDEX, 1);
        let registered = registered_node(identifier.as_str()).expect("pinned Root entry");
        assert_eq!(registered.metadata.display_name, "Root");
        assert_eq!(registered.metadata.category, "Math: Arithmetic");
        assert_eq!(registered.metadata.properties, None);
        assert_eq!(registered.metadata.fields.len(), 2);
        for (field, name) in registered
            .metadata
            .fields
            .iter()
            .zip(["Radicand", "Degree"])
        {
            assert_eq!(field.name, name);
            assert_eq!(field.default_type.as_ref(), Some(&concrete!(f64)));
            assert!(matches!(
                field.widget_override,
                RegistryWidgetOverride::None
            ));
            assert!(matches!(
                field.value_source,
                RegistryValueSource::Default("2.")
            ));
        }
        let expected = HashSet::from([
            binary_signature(item!(f64), item!(f64), concrete!(Item<f64>)),
            binary_signature(item!(DVec2), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(item!(f64), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(item!(DVec2), item!(f64), concrete!(Item<DVec2>)),
            binary_signature(list!(f64), list!(f64), concrete!(List<f64>)),
            binary_signature(list!(DVec2), list!(DVec2), concrete!(List<DVec2>)),
            binary_signature(list!(f64), list!(DVec2), concrete!(List<DVec2>)),
            binary_signature(list!(DVec2), list!(f64), concrete!(List<DVec2>)),
        ]);
        assert_eq!(registered.signatures.len(), expected.len());
        assert_eq!(
            registered.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
    }

    #[test]
    fn at_gra_node_087_1_subtract_has_exact_native_ports_and_signatures() {
        let identifier = subtract::IDENTIFIER;
        assert_eq!(identifier.as_str(), "math_nodes::SubtractNode");
        assert_eq!(subtract::MinuendInput::NODE_IDENTIFIER, identifier);
        assert_eq!(subtract::MinuendInput::INDEX, 0);
        assert_eq!(subtract::SubtrahendInput::NODE_IDENTIFIER, identifier);
        assert_eq!(subtract::SubtrahendInput::INDEX, 1);
        let registered = registered_node(identifier.as_str()).expect("native Subtract entry");
        assert_eq!(registered.metadata.display_name, "Subtract");
        assert_eq!(registered.metadata.category, "Math: Arithmetic");
        assert_eq!(registered.metadata.properties, None);
        assert_eq!(registered.metadata.fields.len(), 2);
        for (field, (name, default_type)) in registered
            .metadata
            .fields
            .iter()
            .zip([("Minuend", list!(f64)), ("Subtrahend", concrete!(f64))])
        {
            assert_eq!(field.name, name);
            assert_eq!(field.default_type.as_ref(), Some(&default_type));
            assert!(matches!(
                field.widget_override,
                RegistryWidgetOverride::None
            ));
            assert!(matches!(field.value_source, RegistryValueSource::None));
        }
        let expected = HashSet::from([
            binary_signature(item!(f64), item!(f64), concrete!(Item<f64>)),
            binary_signature(item!(i64), item!(i64), concrete!(Item<i64>)),
            binary_signature(item!(DVec2), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(item!(f64), item!(DVec2), concrete!(Item<DVec2>)),
            binary_signature(item!(DVec2), item!(f64), concrete!(Item<DVec2>)),
            binary_signature(list!(f64), list!(f64), concrete!(List<f64>)),
            binary_signature(list!(i64), list!(i64), concrete!(List<i64>)),
            binary_signature(list!(DVec2), list!(DVec2), concrete!(List<DVec2>)),
            binary_signature(list!(f64), list!(DVec2), concrete!(List<DVec2>)),
            binary_signature(list!(DVec2), list!(f64), concrete!(List<DVec2>)),
        ]);
        assert_eq!(registered.signatures.len(), expected.len());
        assert_eq!(
            registered.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
    }

    #[test]
    fn at_gra_node_152_1_registry_uses_original_canonical_metadata_and_input_symbol() {
        let identifier = invert::IDENTIFIER;
        assert_eq!(identifier.as_str(), "raster_nodes::adjustments::InvertNode");
        assert_eq!(invert::InputInput::NODE_IDENTIFIER, identifier);
        assert_eq!(invert::InputInput::INDEX, 0);
        let registered =
            registered_node(identifier.as_str()).expect("original native registry entry");
        assert_eq!(registered.metadata.display_name, "Invert");
        assert_eq!(registered.metadata.category, "Raster: Adjustment");
        assert_eq!(registered.metadata.properties, None);
        assert_eq!(registered.metadata.fields.len(), 1);
        let input = registered
            .metadata
            .fields
            .first()
            .expect("one primary input");
        assert_eq!(input.name, "Input");
        assert_eq!(
            input.default_type,
            Some(graphite_core_types::list!(Raster<CPU>))
        );
        assert!(matches!(
            input.widget_override,
            RegistryWidgetOverride::None
        ));
        assert!(matches!(input.value_source, RegistryValueSource::None));
        assert_eq!(input.default_colors, None);
    }

    #[test]
    fn at_gra_node_152_1_registry_has_all_six_original_native_signatures() {
        let registered =
            registered_node(invert::IDENTIFIER.as_str()).expect("native Invert signatures");
        let expected = HashSet::from([
            item_signature::<Raster<CPU>>(),
            list_signature::<Raster<CPU>>(),
            item_signature::<Color>(),
            list_signature::<Color>(),
            item_signature::<Gradient>(),
            list_signature::<Gradient>(),
        ]);
        assert_eq!(registered.signatures.len(), 6);
        let actual: HashSet<_> = registered.signatures.into_iter().collect();
        assert_eq!(actual.len(), 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn at_gra_node_152_1_registry_clones_are_independent_of_source_tables() {
        let source_identifier = invert::IDENTIFIER;
        let identifier = source_identifier.as_str();
        let before = registered_node(identifier).expect("source entry before owned mutation");
        let mut owned = registered_node(identifier).expect("independent owned entry");
        owned.metadata.fields.clear();
        owned.metadata.display_name = "owned clone only";
        owned.signatures.clear();
        let after = registered_node(identifier).expect("source entry after owned mutation");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
        assert_eq!(after.metadata.fields.len(), 1);
        assert_eq!(after.signatures.len(), 6);
    }

    #[test]
    fn at_gra_node_152_1_missing_identifier_does_not_mutate_original_tables() {
        let metadata_keys: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata is not poisoned")
            .keys()
            .cloned()
            .collect();
        let constructor_keys: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors are not poisoned")
            .keys()
            .cloned()
            .collect();
        let before = registered_node(invert::IDENTIFIER.as_str())
            .expect("source entry before missing lookup");
        for identifier in ["graphics_core::missing_original_node", "Invert"] {
            assert!(matches!(
                registered_node(identifier),
                Err(RegistryError::MissingSourceNode)
            ));
        }
        let metadata_after: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata remains unpoisoned")
            .keys()
            .cloned()
            .collect();
        let constructors_after: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors remain unpoisoned")
            .keys()
            .cloned()
            .collect();
        assert_eq!(metadata_after, metadata_keys);
        assert_eq!(constructors_after, constructor_keys);
        let after = registered_node(invert::IDENTIFIER.as_str())
            .expect("source entry after missing lookup");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
    }

    #[test]
    fn at_gra_node_164_1_registry_uses_original_make_opaque_metadata_and_input() {
        let identifier = make_opaque::IDENTIFIER;
        assert_eq!(
            identifier.as_str(),
            "raster_nodes::adjustments::MakeOpaqueNode"
        );
        assert_eq!(make_opaque::InputInput::NODE_IDENTIFIER, identifier);
        assert_eq!(make_opaque::InputInput::INDEX, 0);
        let registered =
            registered_node(identifier.as_str()).expect("original native Make Opaque entry");
        assert_eq!(registered.metadata.display_name, "Make Opaque");
        assert_eq!(registered.metadata.category, "Raster: Channels");
        assert_eq!(registered.metadata.properties, None);
        assert!(registered.metadata.context_features.is_empty());
        assert!(!registered.metadata.memoize);
        assert!(!registered.metadata.inject_scope);
        assert_eq!(registered.metadata.fields.len(), 1);
        let input = registered
            .metadata
            .fields
            .first()
            .expect("one original Input field");
        assert_eq!(input.name, "Input");
        // Raw metadata reflects #[expose]; node_inputs separately exposes typed
        // primary inputs when constructing the editor document template.
        assert!(!input.exposed);
        assert!(!input.hidden);
        assert_eq!(
            input.default_type,
            Some(graphite_core_types::list!(Raster<CPU>))
        );
        assert!(matches!(
            input.widget_override,
            RegistryWidgetOverride::None
        ));
        assert!(matches!(input.value_source, RegistryValueSource::None));
        assert_eq!(input.default_colors, None);
    }

    #[test]
    fn at_gra_node_164_1_registry_has_all_six_raw_original_make_opaque_signatures() {
        let registered = registered_node(make_opaque::IDENTIFIER.as_str())
            .expect("native Make Opaque signatures");
        let expected = HashSet::from([
            item_signature::<Raster<CPU>>(),
            list_signature::<Raster<CPU>>(),
            item_signature::<Color>(),
            list_signature::<Color>(),
            item_signature::<Gradient>(),
            list_signature::<Gradient>(),
        ]);
        assert_eq!(registered.signatures.len(), 6);
        let actual: HashSet<_> = registered.signatures.into_iter().collect();
        assert_eq!(actual.len(), 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn at_gra_node_164_1_registry_clone_and_missing_labels_preserve_source_entry() {
        let source_identifier = make_opaque::IDENTIFIER;
        let identifier = source_identifier.as_str();
        let before = registered_node(identifier).expect("original Make Opaque source entry");
        let mut owned = registered_node(identifier).expect("owned registry projection");
        owned.metadata.display_name = "only a projected clone";
        owned.metadata.fields.clear();
        owned.signatures.clear();
        for identifier in [
            "Make Opaque",
            "make_opaque",
            "graphics_core::missing_make_opaque",
        ] {
            assert!(matches!(
                registered_node(identifier),
                Err(RegistryError::MissingSourceNode)
            ));
        }
        let after = registered_node(identifier).expect("unchanged original Make Opaque entry");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
        assert_eq!(after.metadata.fields.len(), 1);
        assert_eq!(after.signatures.len(), 6);
    }
    #[test]
    fn at_gra_node_164_1_std_native_gpu_absence_preserves_original_registry() {
        let source_identifier = make_opaque::IDENTIFIER;
        let identifier = source_identifier.as_str();
        let before = registered_node(identifier).expect("original native Make Opaque entry");
        let metadata_keys: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata is not poisoned")
            .keys()
            .cloned()
            .collect();
        let constructor_keys: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors are not poisoned")
            .keys()
            .cloned()
            .collect();
        // The original macro gates this separate module behind shader-nodes;
        // the admitted native dependency uses only default, serde, and std.
        assert!(matches!(
            registered_node(
                "raster_nodes::adjustments::make_opaque_shader_node::MakeOpaqueShaderNodeNode"
            ),
            Err(RegistryError::MissingSourceNode)
        ));
        let after = registered_node(identifier).expect("native entry remains addressable");
        assert_eq!(identifier, "raster_nodes::adjustments::MakeOpaqueNode");
        assert_eq!(after.metadata.display_name, "Make Opaque");
        assert_eq!(after.metadata.category, "Raster: Channels");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
        let expected = HashSet::from([
            item_signature::<Raster<CPU>>(),
            list_signature::<Raster<CPU>>(),
            item_signature::<Color>(),
            list_signature::<Color>(),
            item_signature::<Gradient>(),
            list_signature::<Gradient>(),
        ]);
        assert_eq!(after.signatures.len(), 6);
        assert_eq!(
            after.signatures.into_iter().collect::<HashSet<_>>(),
            expected
        );
        let metadata_after: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata remains unpoisoned")
            .keys()
            .cloned()
            .collect();
        let constructors_after: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors remain unpoisoned")
            .keys()
            .cloned()
            .collect();
        assert_eq!(metadata_after, metadata_keys);
        assert_eq!(constructors_after, constructor_keys);
    }

    fn mix_item_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(Item<T>))),
            vec![
                fn_type_fut!(Context<'static>, Item<T>),
                fn_type_fut!(Context<'static>, Item<T>),
                fn_type_fut!(
                    Context<'static>,
                    Item<graphite_core_types::blending::BlendMode>
                ),
                fn_type_fut!(Context<'static>, Item<f64>),
            ],
        )
    }

    fn mix_list_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(List<T>))),
            vec![
                fn_type_fut!(Context<'static>, List<T>),
                fn_type_fut!(Context<'static>, List<T>),
                fn_type_fut!(
                    Context<'static>,
                    List<graphite_core_types::blending::BlendMode>
                ),
                fn_type_fut!(Context<'static>, List<f64>),
            ],
        )
    }

    #[test]
    fn at_gra_node_143_1_mix_original_identifier_ports_and_metadata() {
        use graphite_raster_nodes::blending_nodes::mix;
        let identifier = mix::IDENTIFIER;
        assert_eq!(identifier.as_str(), "raster_nodes::blending_nodes::MixNode");
        assert_eq!(mix::OverInput::NODE_IDENTIFIER, identifier);
        assert_eq!(mix::UnderInput::NODE_IDENTIFIER, identifier);
        assert_eq!(mix::BlendModeInput::NODE_IDENTIFIER, identifier);
        assert_eq!(mix::OpacityInput::NODE_IDENTIFIER, identifier);
        assert_eq!(mix::OverInput::INDEX, 0);
        assert_eq!(mix::UnderInput::INDEX, 1);
        assert_eq!(mix::BlendModeInput::INDEX, 2);
        assert_eq!(mix::OpacityInput::INDEX, 3);
        let registered = registered_node(identifier.as_str()).expect("original native Mix entry");
        assert_eq!(registered.metadata.display_name, "Mix");
        assert_eq!(registered.metadata.category, "Raster");
        assert_eq!(registered.metadata.properties, None);
        assert_eq!(registered.metadata.fields.len(), 4);
        assert_eq!(
            registered
                .metadata
                .fields
                .iter()
                .map(|field| field.name)
                .collect::<Vec<_>>(),
            ["Over", "Under", "Blend Mode", "Opacity"]
        );
    }

    #[test]
    fn at_gra_node_143_1_mix_registry_has_exact_six_four_port_signatures() {
        use graphite_raster_nodes::blending_nodes::mix;
        let registered = registered_node(mix::IDENTIFIER.as_str()).expect("native Mix signatures");
        let expected = HashSet::from([
            mix_item_signature::<Raster<CPU>>(),
            mix_list_signature::<Raster<CPU>>(),
            mix_item_signature::<Color>(),
            mix_list_signature::<Color>(),
            mix_item_signature::<Gradient>(),
            mix_list_signature::<Gradient>(),
        ]);
        assert_eq!(registered.signatures.len(), 6);
        let actual: HashSet<_> = registered.signatures.into_iter().collect();
        assert_eq!(actual.len(), 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn at_gra_node_143_1_mix_lookup_does_not_rebind_display_names_or_mutate_source() {
        use graphite_raster_nodes::blending_nodes::mix;
        let source_identifier = mix::IDENTIFIER;
        let identifier = source_identifier.as_str();
        let before = registered_node(identifier).expect("original source Mix entry");
        let mut owned = registered_node(identifier).expect("owned Mix projection");
        owned.metadata.fields.clear();
        owned.signatures.clear();
        for name in ["Mix", "mix", "graphics_core::missing_mix"] {
            assert!(matches!(
                registered_node(name),
                Err(RegistryError::MissingSourceNode)
            ));
        }
        let after = registered_node(identifier).expect("unchanged source Mix entry");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
    }

    #[test]
    fn at_gra_node_143_1_mix_raw_fields_preserve_source_defaults_exposure_and_numeric_settings() {
        use graphite_core_types::blending::BlendMode;
        use graphite_raster_nodes::blending_nodes::mix;

        let registered =
            registered_node(mix::IDENTIFIER.as_str()).expect("actual native Mix metadata");
        let fields = &registered.metadata.fields;
        assert_eq!(fields.len(), 4);
        let expected = [
            ("Over", graphite_core_types::list!(Raster<CPU>), false),
            ("Under", concrete!(Raster<CPU>), true),
            ("Blend Mode", graphite_core_types::item!(BlendMode), false),
            ("Opacity", concrete!(f64), false),
        ];
        for (field, (name, default_type, exposed)) in fields.iter().zip(expected) {
            assert_eq!(field.name, name);
            assert_eq!(field.default_type, Some(default_type), "{name}");
            assert_eq!(field.exposed, exposed, "{name}");
            assert!(!field.hidden, "{name}");
            assert_eq!(field.description, "", "{name}");
            assert!(
                matches!(field.widget_override, RegistryWidgetOverride::None),
                "{name}"
            );
            assert_eq!(field.default_colors, None, "{name}");
            assert_eq!(field.number_soft_min, None, "{name}");
            assert_eq!(field.number_soft_max, None, "{name}");
            assert_eq!(field.number_display_decimal_places, None, "{name}");
            assert_eq!(field.number_step, None, "{name}");
            assert!(!field.multiline, "{name}");
            assert!(!field.progression, "{name}");
            if name == "Opacity" {
                assert!(matches!(
                    field.value_source,
                    RegistryValueSource::Default("100.")
                ));
                assert_eq!(field.number_hard_min, Some(0.));
                assert_eq!(field.number_hard_max, Some(100.));
                assert!(field.number_mode_range);
                assert_eq!(field.unit, Some("%"));
            } else {
                assert!(
                    matches!(field.value_source, RegistryValueSource::None),
                    "{name}"
                );
                assert_eq!(field.number_hard_min, None, "{name}");
                assert_eq!(field.number_hard_max, None, "{name}");
                assert!(!field.number_mode_range, "{name}");
                assert_eq!(field.unit, None, "{name}");
            }
        }
    }

    fn black_and_white_item_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(Item<T>))),
            vec![
                fn_type_fut!(Context<'static>, Item<T>),
                fn_type_fut!(Context<'static>, Item<bool>),
                fn_type_fut!(Context<'static>, Item<Color>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
            ],
        )
    }

    fn black_and_white_list_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(List<T>))),
            vec![
                fn_type_fut!(Context<'static>, List<T>),
                fn_type_fut!(Context<'static>, List<bool>),
                fn_type_fut!(Context<'static>, List<Color>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
            ],
        )
    }

    #[test]
    fn at_gra_node_144_1_black_and_white_original_identifier_ports_and_metadata() {
        use graphite_raster_nodes::adjustments::black_and_white;

        let identifier = black_and_white::IDENTIFIER;
        assert_eq!(
            identifier.as_str(),
            "raster_nodes::adjustments::BlackAndWhiteNode"
        );
        let ports = [
            (
                black_and_white::ImageInput::NODE_IDENTIFIER,
                black_and_white::ImageInput::INDEX,
            ),
            (
                black_and_white::UseTintInput::NODE_IDENTIFIER,
                black_and_white::UseTintInput::INDEX,
            ),
            (
                black_and_white::TintInput::NODE_IDENTIFIER,
                black_and_white::TintInput::INDEX,
            ),
            (
                black_and_white::RedsInput::NODE_IDENTIFIER,
                black_and_white::RedsInput::INDEX,
            ),
            (
                black_and_white::YellowsInput::NODE_IDENTIFIER,
                black_and_white::YellowsInput::INDEX,
            ),
            (
                black_and_white::GreensInput::NODE_IDENTIFIER,
                black_and_white::GreensInput::INDEX,
            ),
            (
                black_and_white::CyansInput::NODE_IDENTIFIER,
                black_and_white::CyansInput::INDEX,
            ),
            (
                black_and_white::BluesInput::NODE_IDENTIFIER,
                black_and_white::BluesInput::INDEX,
            ),
            (
                black_and_white::MagentasInput::NODE_IDENTIFIER,
                black_and_white::MagentasInput::INDEX,
            ),
        ];
        for (expected_index, (port_identifier, actual_index)) in ports.into_iter().enumerate() {
            assert_eq!(port_identifier, identifier);
            assert_eq!(actual_index, expected_index);
        }
        let registered =
            registered_node(identifier.as_str()).expect("original Black & White entry");
        assert_eq!(registered.metadata.display_name, "Black & White");
        assert_eq!(registered.metadata.category, "Raster: Adjustment");
        assert_eq!(registered.metadata.description, "");
        assert_eq!(
            registered.metadata.properties,
            Some("black_and_white_properties")
        );
        assert!(registered.metadata.context_features.is_empty());
        assert!(!registered.metadata.memoize);
        assert!(!registered.metadata.inject_scope);
        assert_eq!(registered.metadata.fields.len(), 9);
        assert_eq!(
            registered
                .metadata
                .fields
                .iter()
                .map(|field| field.name)
                .collect::<Vec<_>>(),
            [
                "Image", "Use Tint", "Tint", "Reds", "Yellows", "Greens", "Cyans", "Blues",
                "Magentas"
            ]
        );
    }

    #[test]
    fn at_gra_node_144_1_black_and_white_registry_has_exact_six_nine_port_signatures() {
        use graphite_raster_nodes::adjustments::black_and_white;

        let registered = registered_node(black_and_white::IDENTIFIER.as_str())
            .expect("native Black & White signatures");
        let expected = HashSet::from([
            black_and_white_item_signature::<Raster<CPU>>(),
            black_and_white_list_signature::<Raster<CPU>>(),
            black_and_white_item_signature::<Color>(),
            black_and_white_list_signature::<Color>(),
            black_and_white_item_signature::<Gradient>(),
            black_and_white_list_signature::<Gradient>(),
        ]);
        assert_eq!(registered.signatures.len(), 6);
        let actual: HashSet<_> = registered.signatures.into_iter().collect();
        assert_eq!(actual.len(), 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn at_gra_node_144_1_black_and_white_raw_defaults_and_numeric_settings() {
        use graphite_raster_nodes::adjustments::black_and_white;

        let registered = registered_node(black_and_white::IDENTIFIER.as_str())
            .expect("raw Black & White metadata");
        let expected = [
            (
                "Image",
                graphite_core_types::list!(Raster<CPU>),
                None,
                false,
            ),
            ("Use Tint", graphite_core_types::item!(bool), None, false),
            ("Tint", concrete!(Color), Some("#e1d3b3"), false),
            ("Reds", concrete!(f64), Some("40."), true),
            ("Yellows", concrete!(f64), Some("60."), true),
            ("Greens", concrete!(f64), Some("40."), true),
            ("Cyans", concrete!(f64), Some("60."), true),
            ("Blues", concrete!(f64), Some("20."), true),
            ("Magentas", concrete!(f64), Some("80."), true),
        ];
        assert_eq!(registered.metadata.fields.len(), expected.len());
        for (field, (name, default_type, default_value, numeric)) in
            registered.metadata.fields.iter().zip(expected)
        {
            assert_eq!(field.name, name);
            assert_eq!(field.default_type, Some(default_type), "{name}");
            assert!(!field.exposed, "{name}");
            assert!(!field.hidden, "{name}");
            assert_eq!(field.description, "", "{name}");
            assert!(
                matches!(field.widget_override, RegistryWidgetOverride::None),
                "{name}"
            );
            assert!(
                match (default_value, &field.value_source) {
                    (None, RegistryValueSource::None) => true,
                    (Some(expected), RegistryValueSource::Default(actual)) => *actual == expected,
                    _ => false,
                },
                "unexpected raw value source for {name}: {:?}",
                field.value_source
            );
            assert_eq!(field.default_colors, None, "{name}");
            assert_eq!(field.number_soft_min, numeric.then_some(-200.), "{name}");
            assert_eq!(field.number_soft_max, numeric.then_some(300.), "{name}");
            assert_eq!(field.number_hard_min, None, "{name}");
            assert_eq!(field.number_hard_max, None, "{name}");
            assert_eq!(field.number_mode_range, numeric, "{name}");
            assert_eq!(field.unit, numeric.then_some("%"), "{name}");
            assert_eq!(field.number_display_decimal_places, None, "{name}");
            assert_eq!(field.number_step, None, "{name}");
            assert!(!field.multiline, "{name}");
            assert!(!field.progression, "{name}");
        }
    }

    #[test]
    fn at_gra_node_144_1_black_and_white_missing_labels_and_clones_preserve_source_tables() {
        use graphite_raster_nodes::adjustments::black_and_white;

        let identifier = black_and_white::IDENTIFIER;
        let metadata_keys: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata")
            .keys()
            .cloned()
            .collect();
        let constructor_keys: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors")
            .keys()
            .cloned()
            .collect();
        let before = registered_node(identifier.as_str()).expect("original source entry");
        let mut owned = registered_node(identifier.as_str()).expect("owned projection");
        owned.metadata.display_name = "owned projection only";
        owned.metadata.fields.clear();
        owned.signatures.clear();
        for name in [
            "Black & White",
            "black_and_white",
            "graphics_core::missing_black_and_white",
        ] {
            assert!(matches!(
                registered_node(name),
                Err(RegistryError::MissingSourceNode)
            ));
        }
        let after =
            registered_node(identifier.as_str()).expect("original entry after missing lookup");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
        assert_eq!(after.metadata.fields.len(), 9);
        assert_eq!(after.signatures.len(), 6);
        let metadata_after: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata")
            .keys()
            .cloned()
            .collect();
        let constructors_after: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors")
            .keys()
            .cloned()
            .collect();
        assert_eq!(metadata_after, metadata_keys);
        assert_eq!(constructors_after, constructor_keys);
    }

    fn brightness_contrast_item_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(Item<T>))),
            vec![
                fn_type_fut!(Context<'static>, Item<T>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<f64>),
                fn_type_fut!(Context<'static>, Item<bool>),
                fn_type_fut!(Context<'static>, Item<f64>),
            ],
        )
    }

    fn brightness_contrast_list_signature<T: 'static>() -> NodeIOTypes {
        NodeIOTypes::new(
            concrete!(Context<'static>),
            Type::Future(Box::new(concrete!(List<T>))),
            vec![
                fn_type_fut!(Context<'static>, List<T>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<f64>),
                fn_type_fut!(Context<'static>, List<bool>),
                fn_type_fut!(Context<'static>, List<f64>),
            ],
        )
    }

    #[test]
    fn at_gra_node_145_1_brightness_contrast_original_identifier_ports_and_metadata() {
        use graphite_raster_nodes::adjustments::brightness_contrast;

        let identifier = brightness_contrast::IDENTIFIER;
        assert_eq!(
            identifier.as_str(),
            "raster_nodes::adjustments::BrightnessContrastNode"
        );
        let ports = [
            (
                brightness_contrast::InputInput::NODE_IDENTIFIER,
                brightness_contrast::InputInput::INDEX,
            ),
            (
                brightness_contrast::BrightnessInput::NODE_IDENTIFIER,
                brightness_contrast::BrightnessInput::INDEX,
            ),
            (
                brightness_contrast::ContrastInput::NODE_IDENTIFIER,
                brightness_contrast::ContrastInput::INDEX,
            ),
            (
                brightness_contrast::UseClassicInput::NODE_IDENTIFIER,
                brightness_contrast::UseClassicInput::INDEX,
            ),
            (
                brightness_contrast::ClassicPivotInput::NODE_IDENTIFIER,
                brightness_contrast::ClassicPivotInput::INDEX,
            ),
        ];
        for (expected_index, (port_identifier, actual_index)) in ports.into_iter().enumerate() {
            assert_eq!(port_identifier, identifier);
            assert_eq!(actual_index, expected_index);
        }
        let registered =
            registered_node(identifier.as_str()).expect("original Brightness/Contrast entry");
        assert_eq!(registered.metadata.display_name, "Brightness/Contrast");
        assert_eq!(registered.metadata.category, "Raster: Adjustment");
        assert_eq!(registered.metadata.description, "");
        assert_eq!(
            registered.metadata.properties,
            Some("brightness_contrast_properties")
        );
        assert!(registered.metadata.context_features.is_empty());
        assert!(!registered.metadata.memoize);
        assert!(!registered.metadata.inject_scope);
        assert_eq!(
            registered
                .metadata
                .fields
                .iter()
                .map(|field| field.name)
                .collect::<Vec<_>>(),
            [
                "Input",
                "Brightness",
                "Contrast",
                "Use Classic",
                "Classic Pivot"
            ]
        );
    }

    #[test]
    fn at_gra_node_145_1_brightness_contrast_registry_has_exact_six_five_port_signatures() {
        use graphite_raster_nodes::adjustments::brightness_contrast;

        let registered = registered_node(brightness_contrast::IDENTIFIER.as_str())
            .expect("native Brightness/Contrast signatures");
        let expected = HashSet::from([
            brightness_contrast_item_signature::<Raster<CPU>>(),
            brightness_contrast_list_signature::<Raster<CPU>>(),
            brightness_contrast_item_signature::<Color>(),
            brightness_contrast_list_signature::<Color>(),
            brightness_contrast_item_signature::<Gradient>(),
            brightness_contrast_list_signature::<Gradient>(),
        ]);
        assert_eq!(registered.signatures.len(), 6);
        let actual: HashSet<_> = registered.signatures.into_iter().collect();
        assert_eq!(actual.len(), 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn at_gra_node_145_1_brightness_contrast_raw_defaults_and_numeric_settings() {
        use graphite_raster_nodes::adjustments::brightness_contrast;

        let registered = registered_node(brightness_contrast::IDENTIFIER.as_str())
            .expect("raw Brightness/Contrast metadata");
        let expected = [
            ("Input", graphite_core_types::list!(Raster<CPU>), None),
            ("Brightness", graphite_core_types::item!(f64), None),
            ("Contrast", graphite_core_types::item!(f64), None),
            ("Use Classic", graphite_core_types::item!(bool), None),
            ("Classic Pivot", concrete!(f64), Some("127.")),
        ];
        assert_eq!(registered.metadata.fields.len(), expected.len());
        for (field, (name, default_type, default_value)) in
            registered.metadata.fields.iter().zip(expected)
        {
            assert_eq!(field.name, name);
            assert_eq!(field.default_type, Some(default_type), "{name}");
            assert!(!field.exposed, "{name}");
            assert!(!field.hidden, "{name}");
            assert_eq!(field.description, "", "{name}");
            assert!(
                matches!(field.widget_override, RegistryWidgetOverride::None),
                "{name}"
            );
            assert!(
                match (default_value, &field.value_source) {
                    (None, RegistryValueSource::None) => true,
                    (Some(expected), RegistryValueSource::Default(actual)) => *actual == expected,
                    _ => false,
                },
                "unexpected raw value source for {name}: {:?}",
                field.value_source
            );
            let (hard_min, hard_max, soft_min, soft_max, range, unit) = match name {
                "Brightness" => (Some(-150.), Some(150.), None, None, true, Some("%")),
                "Contrast" => (Some(-100.), Some(100.), None, None, true, Some("%")),
                "Classic Pivot" => (None, None, Some(0.), Some(255.), true, None),
                _ => (None, None, None, None, false, None),
            };
            assert_eq!(field.number_hard_min, hard_min, "{name}");
            assert_eq!(field.number_hard_max, hard_max, "{name}");
            assert_eq!(field.number_soft_min, soft_min, "{name}");
            assert_eq!(field.number_soft_max, soft_max, "{name}");
            assert_eq!(field.number_mode_range, range, "{name}");
            assert_eq!(field.unit, unit, "{name}");
            assert_eq!(field.default_colors, None, "{name}");
            assert_eq!(field.number_display_decimal_places, None, "{name}");
            assert_eq!(field.number_step, None, "{name}");
            assert!(!field.multiline, "{name}");
            assert!(!field.progression, "{name}");
        }
    }

    #[test]
    fn at_gra_node_145_1_brightness_contrast_rejects_display_names_without_mutating_tables() {
        use graphite_raster_nodes::adjustments::brightness_contrast;

        let identifier = brightness_contrast::IDENTIFIER;
        let metadata_keys: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata")
            .keys()
            .cloned()
            .collect();
        let constructor_keys: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors")
            .keys()
            .cloned()
            .collect();
        let before = registered_node(identifier.as_str()).expect("original source entry");
        let mut owned = registered_node(identifier.as_str()).expect("owned projection");
        owned.metadata.display_name = "owned projection only";
        owned.metadata.fields.clear();
        owned.signatures.clear();
        for name in [
            "Brightness/Contrast",
            "brightness_contrast",
            "graphics_core::missing_brightness_contrast",
        ] {
            assert!(matches!(
                registered_node(name),
                Err(RegistryError::MissingSourceNode)
            ));
        }
        let after =
            registered_node(identifier.as_str()).expect("original entry after missing lookup");
        assert_eq!(
            format!("{:?}", before.metadata),
            format!("{:?}", after.metadata)
        );
        assert_eq!(before.signatures, after.signatures);
        let metadata_after: HashSet<_> = NODE_METADATA
            .lock()
            .expect("source metadata")
            .keys()
            .cloned()
            .collect();
        let constructors_after: HashSet<_> = NODE_REGISTRY
            .lock()
            .expect("source constructors")
            .keys()
            .cloned()
            .collect();
        assert_eq!(metadata_after, metadata_keys);
        assert_eq!(constructors_after, constructor_keys);
    }
}
