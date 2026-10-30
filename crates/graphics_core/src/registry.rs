use std::fmt;

use graphite_core_types::{
    NodeIOTypes, ProtoNodeIdentifier,
    registry::{NODE_METADATA, NODE_REGISTRY, NodeMetadata},
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use graphite_core_types::{
        Color, Context, NodeParameter, Type, concrete, fn_type_fut,
        list::{Item, List},
        registry::{RegistryValueSource, RegistryWidgetOverride},
    };
    use graphite_raster_nodes::adjustments::{invert, make_opaque};
    use graphite_raster_types::{CPU, Raster};
    use graphite_vector_types::Gradient;

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
