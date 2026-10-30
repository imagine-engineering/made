pub mod descriptor;
pub mod execution;
pub mod object_info;
pub mod provider_contracts;
pub mod registry_generator;
pub mod slice_registry;
pub mod source_type;
pub mod stored_payload;
pub mod text_format;
pub mod text_regex;

pub use descriptor::{
    CatalogNodeDescriptor, CatalogNodeInputSchemaMetadata, CatalogNodeOutputSchemaMetadata,
    CatalogNodeSchemaMetadata, CatalogNodeSource, CatalogNodeStatus,
    NATIVE_SCHEMA_METADATA_VERSION, NODE_DESCRIPTOR_SCHEMA_VERSION, NativeDescriptorSchemaMetadata,
    NativeDynamicSchemaMetadata, NativeInputRequirement, NativeInputSchemaMetadata,
    NativeNodeSchemaMetadata, NativeOutputSchemaMetadata, NativeSchemaError, NativeSchemaField,
    NativeSchemaProvenance, NativeSchemaValue, NativeSourcePresentationMetadata,
    NativeStructuredInputField, NativeStructuredInputOption, NativeUploadKind, NodeDescriptor,
    PortDescriptor,
};
pub use execution::{
    LEGACY_NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
    NATIVE_OPAQUE_HANDLE_SCHEMA_VERSION, NATIVE_STRUCTURED_VALUE_SCHEMA_VERSION,
    NATIVE_TEXT_GENERATION_MAX_NEW_TOKENS, NATIVE_TEXT_GENERATION_RNG_PHASE, NativeAssetNameList,
    NativeAssetNameListRequest, NativeAssetNameResolution, NativeAssetNameResolutionRequest,
    NativeAssetReadRequest, NativeAssetReference, NativeAssetResolver, NativeAssetServiceError,
    NativeCacheDependencies, NativeCachePolicy, NativeCompiledModelInvocation,
    NativeCompiledModelInvocationError, NativeComponentH264Mp4BackingRequest,
    NativeComponentH264Mp4BackingService, NativeComponentH264Mp4BackingServiceError,
    NativeComponentH264Mp4BackingServiceIdentity, NativeDynamicInputDescriptor, NativeEffectClass,
    NativeEffectServiceError, NativeEncodedWebm, NativeHandleKind, NativeHandleStore,
    NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType, NativeImagePreviewError,
    NativeInputDescriptor, NativeLtxvPreprocessService, NativeLtxvPreprocessServiceError,
    NativeLtxvPreprocessServiceIdentity, NativeModelCompileBackend, NativeModelCompileOutput,
    NativeModelCompileRequest, NativeModelCompileService, NativeModelCompileServiceError,
    NativeModelResourceRequest, NativeModelResourceService, NativeModelResourceServiceError,
    NativeModelTrainingAlgorithm, NativeModelTrainingOutput, NativeModelTrainingRequest,
    NativeModelTrainingService, NativeModelTrainingServiceError,
    NativeMultimodalTextGenerationRequest, NativeNamedAssetReference, NativeNode,
    NativeNodeBinding, NativeNodeBindingDisposition, NativeNodeBindingsFactory,
    NativeNodeComputeSession, NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation,
    NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativeOutputDescriptor,
    NativeOutputEffectRequest, NativeOutputMediaKind, NativeOutputNamespace, NativeOutputShape,
    NativePortCardinality, NativePreparedEffectKind, NativePreparedEffectRequest,
    NativePreparedEffectService, NativePreparedImagePreview, NativePrimitive, NativePrimitiveType,
    NativeProviderExecutionIdentity, NativeResolvedAsset, NativeResolvedPayload,
    NativeResolvedPayloadRetention, NativeSdPoseRequest, NativeSdPoseService,
    NativeStructuredValue, NativeTextGenerationService, NativeTextGenerationServiceError,
    NativeTextGenerationServiceOutput, NativeTypeUnion, NativeValue, NativeValueType,
    NativeWebmEncodeRequest, NativeWebmEncodeService, NativeWebmEncodeServiceError,
    NativeWebmEncodeServiceIdentity, native_text_generation_transaction,
    native_value_matches_input_schema, native_value_matches_source_input_schema,
    validate_generated_family_bindings,
};
pub use object_info::{
    OBJECT_INFO_SCHEMA_VERSION, ObjectInfoInputSchema, ObjectInfoNode, ObjectInfoOutputSchema,
    ObjectInfoRegistry,
};
pub use provider_contracts::{
    PROVIDER_COMPONENT_CONTRACT_CATALOG, PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256,
    PROVIDER_NAMESPACE_COUNT, PROVIDER_NODE_CONTRACT_COUNT, ProviderContractError,
    ProviderNamespaceProjection, ProviderNodeContract, authoritative_provider_namespace,
    authoritative_provider_namespace_projection, project_authoritative_provider_bindings,
    validate_provider_component_catalog,
};
pub use registry_generator::{
    INACTIVE_NODE_CATALOG, NODE_CONTRACT_CATALOG, NodeRegistry, NodeRegistryError,
    NodeRegistryGenerator, REGISTERED_NODE_CATALOG, built_in_source_schema,
};
pub use slice_registry::{
    DIFFUSION_SLICE_NODE_IDS, EarlySliceRegistry, IMAGE_SLICE_NODE_IDS, SliceRegistryError,
};
pub use source_type::{
    NativeSourceTypeError, NativeSourceTypeOwner, NativeSourceTypeProjection,
    NativeSourceValueClass, native_custom_source_type_projection, native_handle_type_accepts,
    native_plugin_source_type_projection, native_source_type_projection,
    native_value_type_for_output_schema, native_value_types_for_input_schema,
};
pub use stored_payload::{
    NativePayloadResidency, NativeProviderPayload, NativeResidentAllocation,
    NativeResidentAllocationId, NativeResidentPayloadKind, NativeStoredGuiderPayload,
    NativeStoredModelPayload, NativeStoredPayload, NativeStoredPayloadError,
};
pub use text_format::{
    NATIVE_TEXT_FORMAT_MAX_RESULT_BYTES, NATIVE_TEXT_FORMAT_MAX_TEMPLATE_BYTES,
    NativeTextFormatError, NativeTextFormatter,
};
pub use text_regex::{
    NATIVE_TEXT_REGEX_BACKTRACK_LIMIT, NATIVE_TEXT_REGEX_MAX_CAPTURE_BYTES,
    NATIVE_TEXT_REGEX_MAX_INPUT_BYTES, NATIVE_TEXT_REGEX_MAX_MATCHES,
    NATIVE_TEXT_REGEX_MAX_PATTERN_BYTES, NativeTextRegex, NativeTextRegexCaptureRows,
    NativeTextRegexError, NativeTextRegexFlags,
};

include!(concat!(env!("OUT_DIR"), "/generated_modules.rs"));

pub use generated_native_diffusion::{
    NATIVE_DIFFUSION_DESCRIPTOR_SCHEMA_VERSION, native_diffusion_descriptors,
};
pub use generated_native_image::{
    NATIVE_IMAGE_DESCRIPTOR_SCHEMA_VERSION, NativeImageDescriptor, NativeImageDescriptorError,
    NativeImageEffect, NativeImagePort, native_image_descriptors,
};

#[cfg(test)]
mod generated_manifest_tests {
    use super::*;
    use std::{collections::BTreeSet, error::Error};

    #[test]
    fn generated_manifests_are_sorted_unique_and_catalog_backed() -> Result<(), Box<dyn Error>> {
        assert!(GENERATED_MODULES.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            GENERATED_DESCRIPTOR_IDS
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert_eq!(
            GENERATED_DESCRIPTOR_IDS.len(),
            GENERATED_DESCRIPTOR_IDS
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
        assert!(
            GENERATED_FAMILY_MODULES
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(
            GENERATED_FAMILY_DESCRIPTOR_IDS
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        assert!(
            GENERATED_FAMILY_MODULES
                .iter()
                .all(|module| GENERATED_MODULES.contains(module))
        );
        assert!(
            GENERATED_FAMILY_DESCRIPTOR_IDS
                .iter()
                .all(|identifier| GENERATED_DESCRIPTOR_IDS.contains(identifier))
        );
        let family_bindings = generated_family_node_bindings()?;
        assert_eq!(family_bindings.len(), GENERATED_FAMILY_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;
        assert!(
            GENERATED_DESCRIPTOR_IDS
                .iter()
                .all(|identifier| registry.descriptor(identifier).is_some())
        );
        Ok(())
    }
}
