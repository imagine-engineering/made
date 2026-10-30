use crate::{
    AssetAvailability, AssetCollisionPolicy, AssetError, AssetIdentity, AssetNamespace, AssetQuery,
    AssetRecord, AssetService, AuthorizedCapabilities, GraphDocument, GraphError,
    MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES, PublishedSubgraphBlueprint, SharedAssetService,
};
use sim_nodes::{NodeDescriptor, PortDescriptor};
use sim_tensor::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
use thiserror::Error;

pub const SUBGRAPH_BLUEPRINT_TYPE_PREFIX: &str = "SubgraphBlueprint.";
pub const SUBGRAPH_BLUEPRINT_CATEGORY: &str = "Subgraph Blueprints/User";
pub const SUBGRAPH_BLUEPRINT_ASSET_TAG: &str = "sim-subgraph-blueprint-v1";
pub const MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES: usize = 1_024;
pub const MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct SubgraphBlueprintCatalogEntry {
    pub descriptor: NodeDescriptor,
    pub description: String,
    pub category: String,
    pub search_aliases: Vec<String>,
    pub asset: AssetRecord,
    pub blueprint: PublishedSubgraphBlueprint,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubgraphBlueprintDiagnostic {
    pub identity: AssetIdentity,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubgraphBlueprintCatalog {
    entries: BTreeMap<String, SubgraphBlueprintCatalogEntry>,
    diagnostics: Vec<SubgraphBlueprintDiagnostic>,
    asset_identities: BTreeSet<AssetIdentity>,
    asset_byte_sizes: BTreeMap<AssetIdentity, usize>,
}

impl SubgraphBlueprintCatalog {
    pub fn entries(&self) -> &BTreeMap<String, SubgraphBlueprintCatalogEntry> {
        &self.entries
    }

    pub fn descriptor(&self, type_identifier: &str) -> Option<&SubgraphBlueprintCatalogEntry> {
        self.entries.get(type_identifier)
    }

    pub fn diagnostics(&self) -> &[SubgraphBlueprintDiagnostic] {
        &self.diagnostics
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubgraphBlueprintPublication {
    pub entry: SubgraphBlueprintCatalogEntry,
    pub catalog: SubgraphBlueprintCatalog,
}

#[derive(Clone)]
pub struct SubgraphBlueprintLibrary {
    assets: SharedAssetService,
    authorization: AuthorizedCapabilities,
}

impl SubgraphBlueprintLibrary {
    pub fn new(assets: SharedAssetService, authorization: AuthorizedCapabilities) -> Self {
        Self {
            assets,
            authorization,
        }
    }

    pub fn publish(
        &self,
        document: &GraphDocument,
        display_name: &str,
        collision_policy: AssetCollisionPolicy,
        cancellation: &CancellationToken,
    ) -> Result<SubgraphBlueprintPublication, SubgraphBlueprintLibraryError> {
        let blueprint = document.export_selected_subgraph_blueprint(display_name)?;
        let identity = self.identity(&blueprint.metadata.filename)?;
        let blueprint_byte_size = blueprint.workflow_bytes.len();
        if blueprint_byte_size > MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES {
            return Err(SubgraphBlueprintLibraryError::BlueprintByteLimit {
                actual: blueprint_byte_size,
                limit: MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES,
            });
        }
        let tags = BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]);
        let mut assets = self
            .assets
            .lock()
            .map_err(|_| SubgraphBlueprintLibraryError::AssetServiceUnavailable)?;
        let mut catalog = self.reload_from_assets(&assets, cancellation)?;
        validate_projected_catalog(&catalog, &identity, blueprint_byte_size)?;
        let asset = assets.write_exact(
            &identity,
            &blueprint.workflow_bytes,
            tags,
            collision_policy,
            &self.authorization,
            cancellation,
        )?;
        let entry = catalog_entry(asset, blueprint);
        catalog
            .diagnostics
            .retain(|diagnostic| diagnostic.identity != identity);
        catalog
            .entries
            .retain(|_, existing| existing.asset.identity != identity);
        catalog
            .entries
            .insert(entry.descriptor.type_name.clone(), entry.clone());
        catalog.asset_identities.insert(identity.clone());
        catalog
            .asset_byte_sizes
            .insert(identity, blueprint_byte_size);
        Ok(SubgraphBlueprintPublication { entry, catalog })
    }

    pub fn load(
        &self,
        display_name: &str,
        cancellation: &CancellationToken,
    ) -> Result<SubgraphBlueprintCatalogEntry, SubgraphBlueprintLibraryError> {
        let filename = format!("{display_name}.json");
        let identity = self.identity(&filename)?;
        let assets = self
            .assets
            .lock()
            .map_err(|_| SubgraphBlueprintLibraryError::AssetServiceUnavailable)?;
        let asset = assets
            .record(&identity)
            .ok_or_else(|| AssetError::UnknownAsset(identity.clone()))?;
        let bytes = assets.read_verified(
            &identity,
            &self.authorization,
            cancellation,
            MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES as u64,
        )?;
        let blueprint = PublishedSubgraphBlueprint::decode(&filename, &bytes)?;
        Ok(catalog_entry(asset, blueprint))
    }

    pub fn reload(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<SubgraphBlueprintCatalog, SubgraphBlueprintLibraryError> {
        let assets = self
            .assets
            .lock()
            .map_err(|_| SubgraphBlueprintLibraryError::AssetServiceUnavailable)?;
        self.reload_from_assets(&assets, cancellation)
    }

    fn reload_from_assets(
        &self,
        assets: &AssetService,
        cancellation: &CancellationToken,
    ) -> Result<SubgraphBlueprintCatalog, SubgraphBlueprintLibraryError> {
        let mut records = Vec::new();
        let mut offset = 0;
        loop {
            if cancellation.is_cancelled() {
                return Err(SubgraphBlueprintLibraryError::Asset(AssetError::Cancelled));
            }
            let page = assets.list_authorized(
                &AssetQuery {
                    namespace: Some(AssetNamespace::Plugin),
                    required_tags: BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
                    availability: Some(AssetAvailability::Present),
                    offset,
                    ..AssetQuery::default()
                },
                &self.authorization,
            )?;
            records.extend(page.records);
            if records.len() > MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES {
                return Err(SubgraphBlueprintLibraryError::CatalogEntryLimit {
                    actual: records.len(),
                    limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES,
                });
            }
            let Some(next_offset) = page.next_offset else {
                break;
            };
            offset = next_offset;
        }

        let mut catalog = SubgraphBlueprintCatalog {
            asset_identities: records.iter().map(|asset| asset.identity.clone()).collect(),
            ..SubgraphBlueprintCatalog::default()
        };
        for asset in &records {
            let byte_size = usize::try_from(asset.byte_size).map_err(|_| {
                SubgraphBlueprintLibraryError::CatalogByteLimit {
                    actual: usize::MAX,
                    limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES,
                }
            })?;
            catalog
                .asset_byte_sizes
                .insert(asset.identity.clone(), byte_size);
        }
        let retained_bytes = catalog
            .asset_byte_sizes
            .values()
            .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
            .ok_or(SubgraphBlueprintLibraryError::CatalogByteLimit {
                actual: usize::MAX,
                limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES,
            })?;
        if retained_bytes > MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES {
            return Err(SubgraphBlueprintLibraryError::CatalogByteLimit {
                actual: retained_bytes,
                limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES,
            });
        }
        for asset in records {
            if asset.identity.subfolder() != Path::new("subgraphs") {
                continue;
            }
            let Some(filename) = asset.identity.filename().map(ToOwned::to_owned) else {
                catalog.diagnostics.push(SubgraphBlueprintDiagnostic {
                    identity: asset.identity,
                    message: "blueprint asset filename is not UTF-8".to_owned(),
                });
                continue;
            };
            let bytes = match assets.read_verified(
                &asset.identity,
                &self.authorization,
                cancellation,
                MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES as u64,
            ) {
                Ok(bytes) => bytes,
                Err(
                    error @ (AssetError::Cancelled
                    | AssetError::PermissionDenied { .. }
                    | AssetError::ProfileMismatch { .. }),
                ) => {
                    return Err(SubgraphBlueprintLibraryError::Asset(error));
                }
                Err(error) => {
                    catalog.diagnostics.push(SubgraphBlueprintDiagnostic {
                        identity: asset.identity,
                        message: error.to_string(),
                    });
                    continue;
                }
            };
            match PublishedSubgraphBlueprint::decode(&filename, &bytes) {
                Ok(blueprint) => {
                    let entry = catalog_entry(asset, blueprint);
                    catalog
                        .entries
                        .insert(entry.descriptor.type_name.clone(), entry);
                }
                Err(error) => catalog.diagnostics.push(SubgraphBlueprintDiagnostic {
                    identity: asset.identity,
                    message: error.to_string(),
                }),
            }
        }
        Ok(catalog)
    }

    fn identity(&self, filename: &str) -> Result<AssetIdentity, AssetError> {
        AssetIdentity::new(
            self.authorization.profile_id(),
            AssetNamespace::Plugin,
            PathBuf::from("subgraphs").join(filename),
        )
    }
}

fn validate_projected_catalog(
    catalog: &SubgraphBlueprintCatalog,
    identity: &AssetIdentity,
    new_bytes: usize,
) -> Result<(), SubgraphBlueprintLibraryError> {
    let projected_entries =
        catalog.asset_identities.len() + usize::from(!catalog.asset_identities.contains(identity));
    if projected_entries > MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES {
        return Err(SubgraphBlueprintLibraryError::CatalogEntryLimit {
            actual: projected_entries,
            limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES,
        });
    }
    let retained_bytes = catalog
        .asset_byte_sizes
        .values()
        .try_fold(0usize, |total, bytes| total.checked_add(*bytes));
    let projected_bytes = retained_bytes
        .and_then(|total| {
            total.checked_sub(
                catalog
                    .asset_byte_sizes
                    .get(identity)
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .and_then(|total| total.checked_add(new_bytes))
        .ok_or(SubgraphBlueprintLibraryError::CatalogByteLimit {
            actual: usize::MAX,
            limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES,
        })?;
    if projected_bytes > MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES {
        return Err(SubgraphBlueprintLibraryError::CatalogByteLimit {
            actual: projected_bytes,
            limit: MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES,
        });
    }
    Ok(())
}

fn catalog_entry(
    asset: AssetRecord,
    blueprint: PublishedSubgraphBlueprint,
) -> SubgraphBlueprintCatalogEntry {
    let metadata = &blueprint.metadata;
    let descriptor = NodeDescriptor {
        type_name: format!("{SUBGRAPH_BLUEPRINT_TYPE_PREFIX}{}", metadata.display_name),
        display_name: metadata.display_name.clone(),
        inputs: metadata
            .inputs
            .iter()
            .map(|port| PortDescriptor {
                name: port.name.clone(),
                type_name: port.port_type.display_name(),
                required: true,
            })
            .collect(),
        outputs: metadata
            .outputs
            .iter()
            .map(|port| PortDescriptor {
                name: port.name.clone(),
                type_name: port.port_type.display_name(),
                required: true,
            })
            .collect(),
    };
    SubgraphBlueprintCatalogEntry {
        descriptor,
        description: metadata.description.clone(),
        category: SUBGRAPH_BLUEPRINT_CATEGORY.to_owned(),
        search_aliases: metadata.search_aliases.clone(),
        asset,
        blueprint,
    }
}

#[derive(Debug, Error)]
pub enum SubgraphBlueprintLibraryError {
    #[error(transparent)]
    Asset(#[from] AssetError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("canonical asset service is unavailable after a panic")]
    AssetServiceUnavailable,
    #[error("subgraph blueprint is {actual} bytes, exceeding its {limit}-byte limit")]
    BlueprintByteLimit { actual: usize, limit: usize },
    #[error("subgraph blueprint catalog has {actual} assets, exceeding its {limit}-asset limit")]
    CatalogEntryLimit { actual: usize, limit: usize },
    #[error("subgraph blueprint catalog retains {actual} bytes, exceeding its {limit}-byte limit")]
    CatalogByteLimit { actual: usize, limit: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssetRoots, AssetService, GraphIdentifier, GraphLevel, GraphNode, GraphPoint, GraphPort,
        GraphPortType, GraphSelection, SubgraphDefinition, SubgraphPort,
        authorize_native_output_ui, authorize_native_subgraph_library,
    };
    use std::{error::Error, fs, sync::Arc};
    use tempfile::TempDir;

    fn service() -> Result<(TempDir, SharedAssetService), Box<dyn Error>> {
        let directory = tempfile::tempdir()?;
        let mut roots = Vec::new();
        for namespace in [
            AssetNamespace::Input,
            AssetNamespace::Output,
            AssetNamespace::Temporary,
            AssetNamespace::Model,
            AssetNamespace::Plugin,
        ] {
            let path = directory.path().join(namespace.locator_type());
            fs::create_dir_all(&path)?;
            roots.push((namespace, path));
        }
        let roots = AssetRoots::new("profile", roots)?;
        Ok((
            directory,
            Arc::new(std::sync::Mutex::new(AssetService::open(roots)?)),
        ))
    }

    fn source_document(description: &str) -> GraphDocument {
        let definition_identifier = GraphIdentifier::from("definition");
        let internal_identifier = GraphIdentifier::from("internal");
        let mut internal = GraphNode::new(
            internal_identifier.clone(),
            "Fixture",
            "Fixture",
            GraphPoint::ZERO,
        );
        internal.inputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        internal.outputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        let port = |identifier: &str| SubgraphPort {
            identifier: identifier.to_owned(),
            name: "image".to_owned(),
            port_type: GraphPortType::Concrete("IMAGE".to_owned()),
            internal_node: Some(internal_identifier.clone()),
            internal_slot: 0,
            source_fields: Default::default(),
        };
        let definition = SubgraphDefinition {
            identifier: definition_identifier.clone(),
            name: "Source".to_owned(),
            graph: Box::new(GraphLevel {
                nodes: BTreeMap::from([(internal_identifier.clone(), internal)]),
                ..GraphLevel::default()
            }),
            inputs: vec![port("input")],
            outputs: vec![port("output")],
            published: false,
            description: description.to_owned(),
            search_aliases: vec!["native".to_owned()],
            exposed_widgets: Vec::new(),
            graph_inline: false,
            unknown: BTreeMap::new(),
        };
        let instance_identifier = GraphIdentifier::from("instance");
        let mut instance = GraphNode::new(
            instance_identifier.clone(),
            definition_identifier.text(),
            "Suggested name",
            GraphPoint::ZERO,
        );
        instance.subgraph_definition = Some(definition_identifier.clone());
        instance.inputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        instance.outputs.push(GraphPort::new(
            "image",
            GraphPortType::Concrete("IMAGE".to_owned()),
        ));
        let mut document = GraphDocument::default();
        document
            .root
            .nodes
            .insert(instance_identifier.clone(), instance);
        document
            .root
            .definitions
            .insert(definition_identifier, definition);
        document.root.selection = GraphSelection {
            nodes: BTreeSet::from([instance_identifier]),
            ..GraphSelection::default()
        };
        document
    }

    #[test]
    fn publish_uses_canonical_asset_transaction_and_registers_exact_descriptor()
    -> Result<(), Box<dyn Error>> {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets, authorization);
        let cancellation = CancellationToken::default();
        let first = library.publish(
            &source_document("First description"),
            "Native Blend",
            AssetCollisionPolicy::Reject,
            &cancellation,
        )?;
        assert_eq!(
            first.entry.descriptor.type_name,
            "SubgraphBlueprint.Native Blend"
        );
        assert_eq!(first.entry.descriptor.inputs[0].type_name, "IMAGE");
        assert_eq!(first.entry.description, "First description");
        assert_eq!(first.catalog.entries().len(), 1);
        assert!(matches!(
            library.publish(
                &source_document("Second description"),
                "Native Blend",
                AssetCollisionPolicy::Reject,
                &cancellation,
            ),
            Err(SubgraphBlueprintLibraryError::Asset(
                AssetError::AlreadyExists(_)
            ))
        ));
        let replacement = library.publish(
            &source_document("Second description"),
            "Native Blend",
            AssetCollisionPolicy::Replace,
            &cancellation,
        )?;
        assert_eq!(replacement.entry.description, "Second description");
        assert_eq!(
            library.load("Native Blend", &cancellation)?.asset.sha256,
            replacement.entry.asset.sha256
        );
        Ok(())
    }

    #[test]
    fn reload_isolates_malformed_assets_and_delegates_authorization() -> Result<(), Box<dyn Error>>
    {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization.clone());
        let cancellation = CancellationToken::default();
        library.publish(
            &source_document("Valid"),
            "Valid",
            AssetCollisionPolicy::Reject,
            &cancellation,
        )?;
        let malformed_identity =
            AssetIdentity::new("profile", AssetNamespace::Plugin, "subgraphs/Broken.json")?;
        assets
            .lock()
            .map_err(|_| "asset service unavailable")?
            .write_exact(
                &malformed_identity,
                b"{not-json",
                BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
                AssetCollisionPolicy::Reject,
                &authorization,
                &cancellation,
            )?;
        let catalog = library.reload(&cancellation)?;
        assert!(catalog.descriptor("SubgraphBlueprint.Valid").is_some());
        assert_eq!(catalog.diagnostics().len(), 1);
        assert_eq!(catalog.diagnostics()[0].identity, malformed_identity);

        let replacement = library.publish(
            &source_document("Repaired"),
            "Broken",
            AssetCollisionPolicy::Replace,
            &cancellation,
        )?;
        assert!(replacement.catalog.diagnostics().is_empty());
        assert_eq!(
            replacement
                .catalog
                .descriptor("SubgraphBlueprint.Broken")
                .map(|entry| entry.description.as_str()),
            Some("Repaired")
        );

        let unauthorized =
            SubgraphBlueprintLibrary::new(assets, authorize_native_output_ui("profile")?);
        assert!(matches!(
            unauthorized.reload(&cancellation),
            Err(SubgraphBlueprintLibraryError::Asset(
                AssetError::PermissionDenied {
                    namespace: AssetNamespace::Plugin,
                    ..
                }
            ))
        ));
        Ok(())
    }

    #[test]
    fn publish_cancellation_precedes_the_canonical_asset_commit() -> Result<(), Box<dyn Error>> {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization);
        let cancellation = CancellationToken::default();
        cancellation.cancel();

        assert!(matches!(
            library.publish(
                &source_document("Cancelled"),
                "Cancelled",
                AssetCollisionPolicy::Reject,
                &cancellation,
            ),
            Err(SubgraphBlueprintLibraryError::Asset(AssetError::Cancelled))
        ));
        let identity = library.identity("Cancelled.json")?;
        assert!(
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .record(&identity)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn oversized_blueprint_rejects_publication_before_asset_mutation() -> Result<(), Box<dyn Error>>
    {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization);
        let cancellation = CancellationToken::default();
        let description = "x".repeat(MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES / 2 + 1024 * 1024);
        let document = source_document(&description);
        let exported = document.export_selected_subgraph_blueprint("Oversized")?;
        assert!(exported.workflow_bytes.len() > MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES);
        assert!(matches!(
            exported.clipboard.encode(),
            Err(GraphError::ClipboardTooLarge(_))
        ));
        assert!(matches!(
            PublishedSubgraphBlueprint::decode("Oversized.json", &exported.workflow_bytes),
            Err(GraphError::BlueprintTooLarge(actual))
                if actual == exported.workflow_bytes.len()
        ));
        assert!(matches!(
            crate::GraphClipboard::from_published_subgraph_blueprint(
                "Oversized.json",
                &exported.workflow_bytes,
            ),
            Err(GraphError::BlueprintTooLarge(actual))
                if actual == exported.workflow_bytes.len()
        ));

        assert!(matches!(
            library.publish(
                &document,
                "Oversized",
                AssetCollisionPolicy::Reject,
                &cancellation,
            ),
            Err(SubgraphBlueprintLibraryError::BlueprintByteLimit {
                actual,
                limit,
            }) if actual == exported.workflow_bytes.len()
                && limit == MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES
        ));
        let identity = library.identity("Oversized.json")?;
        assert!(
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .record(&identity)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn projected_catalog_limits_fail_before_asset_mutation() -> Result<(), Box<dyn Error>> {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization);
        let cancellation = CancellationToken::default();
        let publication = library.publish(
            &source_document("Bounded"),
            "Bounded",
            AssetCollisionPolicy::Reject,
            &cancellation,
        )?;
        let mut catalog = SubgraphBlueprintCatalog::default();
        for index in 0..MAX_SUBGRAPH_BLUEPRINT_CATALOG_ENTRIES {
            let mut entry = publication.entry.clone();
            entry.descriptor.type_name = format!("SubgraphBlueprint.Bounded-{index}");
            let identity = AssetIdentity::new(
                "profile",
                AssetNamespace::Plugin,
                format!("subgraphs/Bounded-{index}.json"),
            )?;
            entry.asset.identity = identity.clone();
            catalog.asset_identities.insert(identity.clone());
            catalog
                .asset_byte_sizes
                .insert(identity, entry.blueprint.workflow_bytes.len());
            catalog
                .entries
                .insert(entry.descriptor.type_name.clone(), entry);
        }
        let overflow_identity =
            AssetIdentity::new("profile", AssetNamespace::Plugin, "subgraphs/Overflow.json")?;
        assert!(matches!(
            validate_projected_catalog(&catalog, &overflow_identity, 1),
            Err(SubgraphBlueprintLibraryError::CatalogEntryLimit { .. })
        ));
        let replacement_identity = AssetIdentity::new(
            "profile",
            AssetNamespace::Plugin,
            "subgraphs/Bounded-0.json",
        )?;
        assert!(validate_projected_catalog(&catalog, &replacement_identity, 1).is_ok());
        assert!(matches!(
            validate_projected_catalog(
                &SubgraphBlueprintCatalog::default(),
                &overflow_identity,
                MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES + 1,
            ),
            Err(SubgraphBlueprintLibraryError::CatalogByteLimit { .. })
        ));
        assert!(
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .record(&overflow_identity)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn replacement_subtracts_existing_bytes_at_catalog_boundary() -> Result<(), Box<dyn Error>> {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization.clone());
        let cancellation = CancellationToken::default();
        let first = library.publish(
            &source_document("A"),
            "Boundary",
            AssetCollisionPolicy::Reject,
            &cancellation,
        )?;
        let first_size = first.entry.blueprint.workflow_bytes.len();
        let padding_size = MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES
            .checked_sub(first_size)
            .ok_or("published blueprint exceeds aggregate test boundary")?;
        let padding_identity =
            AssetIdentity::new("profile", AssetNamespace::Plugin, "subgraphs/Padding.json")?;
        assets
            .lock()
            .map_err(|_| "asset service unavailable")?
            .write_exact(
                &padding_identity,
                &vec![b' '; padding_size],
                BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
                AssetCollisionPolicy::Reject,
                &authorization,
                &cancellation,
            )?;

        let replacement = library.publish(
            &source_document("B"),
            "Boundary",
            AssetCollisionPolicy::Replace,
            &cancellation,
        )?;
        assert_ne!(replacement.entry.asset.sha256, first.entry.asset.sha256);
        assert_eq!(
            replacement.catalog.asset_byte_sizes.values().sum::<usize>(),
            MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES
        );
        assert_eq!(replacement.catalog.diagnostics().len(), 1);
        assert_eq!(
            replacement.catalog.diagnostics()[0].identity,
            padding_identity
        );
        Ok(())
    }

    #[test]
    fn malformed_catalog_bytes_reject_publication_before_asset_mutation()
    -> Result<(), Box<dyn Error>> {
        let (_directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization.clone());
        let cancellation = CancellationToken::default();
        let malformed_bytes = vec![b' '; MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES];
        for index in
            0..(MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES / MAX_PUBLISHED_SUBGRAPH_BLUEPRINT_BYTES)
        {
            let identity = AssetIdentity::new(
                "profile",
                AssetNamespace::Plugin,
                format!("subgraphs/Malformed-{index}.json"),
            )?;
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .write_exact(
                    &identity,
                    &malformed_bytes,
                    BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
                    AssetCollisionPolicy::Reject,
                    &authorization,
                    &cancellation,
                )?;
        }

        let overflow_identity = library.identity("Overflow.json")?;
        assert!(matches!(
            library.publish(
                &source_document("Overflow"),
                "Overflow",
                AssetCollisionPolicy::Reject,
                &cancellation,
            ),
            Err(SubgraphBlueprintLibraryError::CatalogByteLimit { .. })
        ));
        assert!(
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .record(&overflow_identity)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn every_tagged_asset_byte_counts_before_publication_mutation() -> Result<(), Box<dyn Error>> {
        let (directory, assets) = service()?;
        let authorization = authorize_native_subgraph_library("profile")?;
        let library = SubgraphBlueprintLibrary::new(assets.clone(), authorization.clone());
        let cancellation = CancellationToken::default();
        let unreadable_size = 32 * 1024 * 1024;
        let oversized_size = 17 * 1024 * 1024;
        let misplaced_size = 15 * 1024 * 1024;
        assert_eq!(
            unreadable_size + oversized_size + misplaced_size,
            MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES
        );
        let unreadable_identity = AssetIdentity::new(
            "profile",
            AssetNamespace::Plugin,
            "subgraphs/Unreadable.json",
        )?;
        let oversized_identity = AssetIdentity::new(
            "profile",
            AssetNamespace::Plugin,
            "subgraphs/Oversized.json",
        )?;
        let misplaced_identity =
            AssetIdentity::new("profile", AssetNamespace::Plugin, "misplaced/Bulk.bin")?;
        for (identity, byte_size) in [
            (&unreadable_identity, unreadable_size),
            (&oversized_identity, oversized_size),
            (&misplaced_identity, misplaced_size),
        ] {
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .write_exact(
                    identity,
                    &vec![b' '; byte_size],
                    BTreeSet::from([SUBGRAPH_BLUEPRINT_ASSET_TAG.to_owned()]),
                    AssetCollisionPolicy::Reject,
                    &authorization,
                    &cancellation,
                )?;
        }
        fs::write(
            directory
                .path()
                .join(AssetNamespace::Plugin.locator_type())
                .join("subgraphs/Unreadable.json"),
            vec![b'x'; unreadable_size],
        )?;

        let catalog = library.reload(&cancellation)?;
        assert_eq!(
            catalog.asset_byte_sizes.values().sum::<usize>(),
            MAX_SUBGRAPH_BLUEPRINT_CATALOG_BYTES
        );
        assert!(
            catalog
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.identity == unreadable_identity)
        );
        assert!(
            catalog
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.identity == oversized_identity)
        );
        assert!(catalog.asset_identities.contains(&misplaced_identity));

        let overflow_identity = library.identity("Overflow.json")?;
        assert!(matches!(
            library.publish(
                &source_document("Overflow"),
                "Overflow",
                AssetCollisionPolicy::Reject,
                &cancellation,
            ),
            Err(SubgraphBlueprintLibraryError::CatalogByteLimit { .. })
        ));
        assert!(
            assets
                .lock()
                .map_err(|_| "asset service unavailable")?
                .record(&overflow_identity)
                .is_none()
        );
        Ok(())
    }
}
