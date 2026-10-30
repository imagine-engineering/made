use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeEffectClass, NativeHandleKind,
    NativeHandleStoreError, NativeHandleType, NativeInputDescriptor, NativeNode, NativeNodeBinding,
    NativeEffectServiceError, NativeInputSchemaMetadata, NativeNodeBindingsFactory,
    NativeNodeContext, NativeNodeContractError, NativeNodeDescriptor, NativeNodeFailure,
    NativeNodeFailureKind, NativeNodeOutcome, NativeNodePresentation, NativeOpaqueHandle,
    NativeOutputDescriptor, NativeOutputEffectRequest, NativeOutputMediaKind,
    NativeOutputNamespace, NativeOutputShape, NativePortCardinality, NativePrimitive,
    NativePrimitiveType, NativeStoredPayload, NativeTypeUnion, NativeValue, NativeValueType,
    NativeWebmEncodeRequest, NativeWebmEncodeServiceError, built_in_source_schema,
};
use sim_media::{
    NativeVideoBitDepth, NativeVideoCodec, NativeVideoCrf, NativeVideoPayload,
    source_rounded_millisecond_frame_rate,
};
use sim_model::FrameInterpolationError;
use sim_tensor::{
    DType, DeviceId, ImageTensor, Layout, MemoryFormatReference, NativeTensorPayload,
    NativeTensorPayloadError, NativeTensorRole, Scalar, TensorError,
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::div_with_context_exact_native,
    generated_storage_dtype_device_01::contiguous_with_context_exact_native,
};
use sim_types::CancellationToken;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Arc};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "CreateVideo",
    "GetVideoComponents",
    "FrameInterpolate",
    "SaveWEBM",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CLASS_TYPE: &str = "CreateVideo";
const FEATURE_ID: &str = "COMFY-NODE-0124";
const IMPLEMENTATION_VERSION: &str = "source-7b8f73c9-v1";
const CACHE_CHANGE_TOKEN: &str = "source-7b8f73c9-video-components-v1";
const COMPONENTS_CLASS_TYPE: &str = "GetVideoComponents";
const COMPONENTS_FEATURE_ID: &str = "COMFY-NODE-0207";
const COMPONENTS_IMPLEMENTATION_VERSION: &str = "source-b2232b2c-v1";
const COMPONENTS_CACHE_CHANGE_TOKEN: &str = "source-b2232b2c-video-components-v1";
const INTERPOLATE_CLASS_TYPE: &str = "FrameInterpolate";
const INTERPOLATE_FEATURE_ID: &str = "COMFY-NODE-0190";
const INTERPOLATE_IMPLEMENTATION_VERSION: &str = "source-e0b9dd6e-v1";
const INTERPOLATE_CACHE_CHANGE_TOKEN: &str = "source-e0b9dd6e-frame-interpolate-v1";
const SAVE_WEBM_CLASS_TYPE: &str = "SaveWEBM";
const SAVE_WEBM_FEATURE_ID: &str = "COMFY-NODE-0602";
const SAVE_WEBM_IMPLEMENTATION_VERSION: &str = "source-55496b10-v1";
const SAVE_WEBM_CACHE_CHANGE_TOKEN: &str = "source-55496b10-save-webm-v1";

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    Ok(vec![
        native_node_binding()?,
        components_node_binding()?,
        interpolate_node_binding()?,
        save_webm_node_binding()?,
    ])
}

fn save_webm_node_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let mut source_schema = built_in_source_schema(SAVE_WEBM_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &[
                "images".to_owned(),
                "filename_prefix".to_owned(),
                "codec".to_owned(),
                "fps".to_owned(),
                "crf".to_owned(),
            ],
            &[],
            &["images".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    source_schema.inputs.extend([
        NativeInputSchemaMetadata::compatibility("prompt", "PROMPT"),
        NativeInputSchemaMetadata::compatibility("extra_pnginfo", "EXTRA_PNGINFO"),
    ]);
    Ok(NativeNodeBinding::Executable {
        feature_id: SAVE_WEBM_FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: SAVE_WEBM_CLASS_TYPE.to_owned(),
            implementation_version: SAVE_WEBM_IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                handle_input("images", image_type()?, true)?,
                primitive_input("filename_prefix", NativePrimitiveType::String, true)?,
                primitive_input("codec", NativePrimitiveType::String, true)?,
                primitive_input("fps", NativePrimitiveType::Number, true)?,
                primitive_input("crf", NativePrimitiveType::Number, true)?,
                hidden_preserved_input("prompt", "PROMPT")?,
                hidden_preserved_input("extra_pnginfo", "EXTRA_PNGINFO")?,
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "images".to_owned(),
                produced_type: NativeValueType::Handle(image_type()?),
                is_list: false,
            }],
            output_node: true,
            effect: NativeEffectClass::WritesArtifact,
            cache: NativeCachePolicy::Never,
        },
        presentation: NativeNodePresentation {
            display_name: "Save WEBM".to_owned(),
            category: "video".to_owned(),
            description: String::new(),
            output_names: vec!["images".to_owned()],
            search_aliases: vec!["export webm".to_owned()],
            is_deprecated: false,
            is_experimental: true,
        },
        node: Arc::new(SaveWebmNode),
    })
}

fn native_node_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let source_schema = built_in_source_schema(CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &[
                "images".to_owned(),
                "fps".to_owned(),
                "audio".to_owned(),
                "bit_depth".to_owned(),
            ],
            &[],
            &["video".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: CLASS_TYPE.to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                handle_input("images", image_type()?, true)?,
                primitive_input("fps", NativePrimitiveType::Number, true)?,
                handle_input("audio", audio_type()?, false)?,
                primitive_input("bit_depth", NativePrimitiveType::Integer, false)?,
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "video".to_owned(),
                produced_type: NativeValueType::Handle(video_type()?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Create Video".to_owned(),
            category: "video".to_owned(),
            description: "Create a video from images.".to_owned(),
            output_names: vec!["video".to_owned()],
            search_aliases: vec!["images to video".to_owned()],
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(CreateVideoNode),
    })
}

fn components_node_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let source_schema = built_in_source_schema(COMPONENTS_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &["video".to_owned()],
            &[],
            &[
                "images".to_owned(),
                "audio".to_owned(),
                "fps".to_owned(),
                "bit_depth".to_owned(),
            ],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: COMPONENTS_FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: COMPONENTS_CLASS_TYPE.to_owned(),
            implementation_version: COMPONENTS_IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![handle_input("video", video_type()?, true)?],
            dynamic_inputs: Vec::new(),
            outputs: vec![
                NativeOutputDescriptor {
                    name: "images".to_owned(),
                    produced_type: NativeValueType::Handle(image_type()?),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "audio".to_owned(),
                    produced_type: NativeValueType::Handle(audio_type()?),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "fps".to_owned(),
                    produced_type: NativeValueType::Primitive(NativePrimitiveType::Number),
                    is_list: false,
                },
                NativeOutputDescriptor {
                    name: "bit_depth".to_owned(),
                    produced_type: NativeValueType::Primitive(NativePrimitiveType::Integer),
                    is_list: false,
                },
            ],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Get Video Components".to_owned(),
            category: "video".to_owned(),
            description:
                "Extracts all components from a video: frames, audio, framerate, and bit depth."
                    .to_owned(),
            output_names: vec![
                "images".to_owned(),
                "audio".to_owned(),
                "fps".to_owned(),
                "bit_depth".to_owned(),
            ],
            search_aliases: vec![
                "extract frames".to_owned(),
                "split video".to_owned(),
                "video to images".to_owned(),
                "demux".to_owned(),
            ],
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(GetVideoComponentsNode),
    })
}

fn interpolate_node_binding() -> Result<NativeNodeBinding, NativeNodeContractError> {
    let source_schema = built_in_source_schema(INTERPOLATE_CLASS_TYPE)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
        .bind_execution_ports(
            &[
                "interp_model".to_owned(),
                "images".to_owned(),
                "multiplier".to_owned(),
            ],
            &[],
            &["images".to_owned()],
        )
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    Ok(NativeNodeBinding::Executable {
        feature_id: INTERPOLATE_FEATURE_ID.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: INTERPOLATE_CLASS_TYPE.to_owned(),
            implementation_version: INTERPOLATE_IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs: vec![
                handle_input("interp_model", interpolation_model_type()?, true)?,
                handle_input("images", image_type()?, true)?,
                primitive_input("multiplier", NativePrimitiveType::Integer, true)?,
            ],
            dynamic_inputs: Vec::new(),
            outputs: vec![NativeOutputDescriptor {
                name: "images".to_owned(),
                produced_type: NativeValueType::Handle(image_type()?),
                is_list: false,
            }],
            output_node: false,
            effect: NativeEffectClass::Pure,
            cache: NativeCachePolicy::InputIdentity,
        },
        presentation: NativeNodePresentation {
            display_name: "Run Frame Interpolation Model".to_owned(),
            category: "video".to_owned(),
            description: String::new(),
            output_names: vec!["images".to_owned()],
            search_aliases: vec![
                "rife".to_owned(),
                "film".to_owned(),
                "frame interpolation".to_owned(),
                "slow motion".to_owned(),
                "interpolate frames".to_owned(),
                "vfi".to_owned(),
            ],
            is_deprecated: false,
            is_experimental: false,
        },
        node: Arc::new(FrameInterpolateNode),
    })
}

fn handle_input(
    name: &str,
    handle_type: NativeHandleType,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::Handle(handle_type)])?,
        required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: false,
    })
}

fn primitive_input(
    name: &str,
    primitive_type: NativePrimitiveType,
    required: bool,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::Primitive(primitive_type)])?,
        required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: true,
    })
}

fn hidden_preserved_input(
    name: &str,
    type_name: &str,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    Ok(NativeInputDescriptor {
        name: name.to_owned(),
        accepted_types: NativeTypeUnion::new([NativeValueType::NamedPreservedUnknown(
            type_name.to_owned(),
        )])?,
        required: false,
        hidden: true,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: true,
    })
}

fn image_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Image, "IMAGE")
}

fn audio_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Audio, "AUDIO")
}

fn video_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Video, "VIDEO")
}

fn interpolation_model_type() -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(NativeHandleKind::Model, "INTERP_MODEL")
}

#[derive(Debug)]
struct CreateVideoNode;

impl NativeNode for CreateVideoNode {
    fn class_type(&self) -> &str {
        CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_cancellation(context)?;
        parse_inputs(available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        parse_inputs(inputs)?;
        Ok(CACHE_CHANGE_TOKEN.to_owned())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_cancellation(&context)?;
            let parsed = parse_inputs(&inputs)?;
            let image_type = image_type().map_err(|error| invalid_inputs(error.to_string()))?;
            let resolved_image = context
                .handle_store()
                .resolve(parsed.images, &image_type, &context.cancellation)
                .map_err(handle_failure)?;
            let NativeStoredPayload::Tensor(image_payload) = resolved_image.as_ref() else {
                return Err(invalid_payload(
                    "IMAGE handle does not contain a tensor payload",
                ));
            };
            if image_payload.role() != NativeTensorRole::Image {
                return Err(invalid_payload("IMAGE handle has the wrong tensor role"));
            }
            let image = image_payload
                .image()
                .ok_or_else(|| invalid_payload("IMAGE handle has no canonical ImageTensor"))?;

            let resolved_audio = match parsed.audio {
                Some(audio) => Some(
                    context
                        .handle_store()
                        .resolve(
                            audio,
                            &audio_type().map_err(|error| invalid_inputs(error.to_string()))?,
                            &context.cancellation,
                        )
                        .map_err(handle_failure)?,
                ),
                None => None,
            };
            let audio = resolved_audio
                .as_ref()
                .map(|resolved| match resolved.as_ref() {
                    NativeStoredPayload::Audio(audio) => Ok(audio.as_ref().clone()),
                    _ => Err(invalid_payload(
                        "AUDIO handle does not contain an audio payload",
                    )),
                })
                .transpose()?;

            check_cancellation(&context)?;
            let video = NativeVideoPayload::checked(
                image.tensor().clone(),
                parsed.frame_rate.0,
                parsed.frame_rate.1,
                parsed.bit_depth,
                audio,
                None,
                BTreeMap::new(),
            )
            .map_err(|error| invalid_payload(error.to_string()))?;
            check_cancellation(&context)?;
            let output_handle = context
                .handle_store()
                .publish(
                    NativeStoredPayload::Video(Arc::new(video)),
                    &context.cancellation,
                )
                .map_err(handle_failure)?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: output_handle,
                }],
                ui: None,
                effects: Vec::new(),
            };
            outcome
                .validate()
                .map_err(|error| invalid_inputs(error.to_string()))?;
            drop(resolved_audio);
            drop(resolved_image);
            Ok(outcome)
        })
    }
}

#[derive(Debug)]
struct GetVideoComponentsNode;

impl NativeNode for GetVideoComponentsNode {
    fn class_type(&self) -> &str {
        COMPONENTS_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        COMPONENTS_IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_components_cancellation(context)?;
        components_input(available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        components_input(inputs)?;
        Ok(COMPONENTS_CACHE_CHANGE_TOKEN.to_owned())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check_components_cancellation(&context)?;
            let video_handle = components_input(&inputs)?;
            let resolved = context
                .handle_store()
                .resolve(
                    video_handle,
                    &video_type().map_err(|error| components_invalid(error.to_string()))?,
                    &context.cancellation,
                )
                .map_err(components_handle_failure)?;
            let NativeStoredPayload::Video(video) = resolved.as_ref() else {
                return Err(components_invalid(
                    "VIDEO handle does not contain a canonical video payload",
                ));
            };
            video
                .validate()
                .map_err(|error| components_invalid(error.to_string()))?;
            let components = video.components().ok_or_else(|| {
                components_invalid(
                    "GetVideoComponents cannot decode an encoded VIDEO backing yet",
                )
            })?;
            let image = project_video_frames(&context, components.frames())?;
            let image_payload = NativeTensorPayload::from_image(NativeTensorRole::Image, image)
                .map_err(|error| components_invalid(error.to_string()))?;
            let audio_payload = components
                .audio()
                .cloned()
                .map(|audio| NativeStoredPayload::Audio(Arc::new(audio)));
            let (frame_rate_numerator, frame_rate_denominator) = video.frame_rate();
            let frame_rate = frame_rate_numerator as f64 / frame_rate_denominator as f64;
            let bit_depth = i64::from(video.bit_depth().bits());
            drop(resolved);

            let mut published = Vec::with_capacity(2);
            let image_handle = publish_component(
                &context,
                NativeStoredPayload::Tensor(Arc::new(image_payload)),
                &mut published,
            )?;
            let audio_value = if let Some(audio_payload) = audio_payload {
                match publish_component(&context, audio_payload, &mut published) {
                    Ok(handle) => NativeValue::Handle { value: handle },
                    Err(error) => {
                        rollback_components(&context, &published)?;
                        return Err(error);
                    }
                }
            } else {
                NativeValue::Primitive {
                    value: NativePrimitive::Null,
                }
            };
            if let Err(error) = check_components_cancellation(&context) {
                rollback_components(&context, &published)?;
                return Err(error);
            }
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![
                    NativeValue::Handle {
                        value: image_handle,
                    },
                    audio_value,
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(frame_rate),
                    },
                    NativeValue::Primitive {
                        value: NativePrimitive::Integer(bit_depth),
                    },
                ],
                ui: None,
                effects: Vec::new(),
            };
            if let Err(error) = outcome.validate() {
                rollback_components(&context, &published)?;
                return Err(components_invalid(error.to_string()));
            }
            Ok(outcome)
        })
    }
}

#[derive(Debug)]
struct FrameInterpolateNode;

impl NativeNode for FrameInterpolateNode {
    fn class_type(&self) -> &str {
        INTERPOLATE_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        INTERPOLATE_IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_interpolation_cancellation(context)?;
        interpolation_inputs(available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        interpolation_inputs(inputs)?;
        Ok(INTERPOLATE_CACHE_CHANGE_TOKEN.to_owned())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move { execute_frame_interpolate(&context, &inputs) })
    }
}

#[derive(Debug)]
struct SaveWebmNode;

impl NativeNode for SaveWebmNode {
    fn class_type(&self) -> &str {
        SAVE_WEBM_CLASS_TYPE
    }

    fn implementation_version(&self) -> &str {
        SAVE_WEBM_IMPLEMENTATION_VERSION
    }

    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        available_inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<std::collections::BTreeSet<String>, NativeNodeFailure> {
        check_save_webm_cancellation(context)?;
        save_webm_inputs(available_inputs)?;
        Ok(std::collections::BTreeSet::new())
    }

    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        save_webm_inputs(inputs)?;
        Ok(SAVE_WEBM_CACHE_CHANGE_TOKEN.to_owned())
    }

    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move { execute_save_webm(&context, &inputs).await })
    }
}

async fn execute_save_webm(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_save_webm_cancellation(context)?;
    let parsed = save_webm_inputs(inputs)?;
    let input_handle = parsed.images.clone();
    let resolved = context
        .handle_store()
        .resolve(
            parsed.images,
            &image_type().map_err(|error| save_webm_invalid(error.to_string()))?,
            &context.cancellation,
        )
        .map_err(save_webm_handle_failure)?;
    let NativeStoredPayload::Tensor(payload) = resolved.as_ref() else {
        return Err(save_webm_invalid(
            "IMAGE handle does not contain a tensor payload",
        ));
    };
    if payload.role() != NativeTensorRole::Image {
        return Err(save_webm_invalid("IMAGE handle has the wrong tensor role"));
    }
    let images = payload
        .image()
        .ok_or_else(|| save_webm_invalid("IMAGE handle has no canonical ImageTensor"))?
        .clone();
    drop(resolved);

    let metadata = save_webm_metadata(parsed.prompt, parsed.extra_pnginfo)?;
    let request = NativeWebmEncodeRequest::checked(
        images,
        parsed.codec,
        parsed.frame_rate,
        parsed.crf,
        metadata,
    )
    .map_err(save_webm_encode_failure)?;
    let compute = context
        .compute_session()
        .map_err(|error| save_webm_service_failure(error.to_string()))?;
    let execution = compute
        .execution_context(context)
        .map_err(|error| save_webm_service_failure(error.to_string()))?;
    let encoded = context
        .webm_encode_service()
        .map_err(save_webm_encode_failure)?
        .encode_webm(request, &execution)
        .await
        .map_err(save_webm_encode_failure)?;
    check_save_webm_cancellation(context)?;

    let encoded_bytes = encoded
        .bytes()
        .contiguous_bytes()
        .map_err(save_webm_tensor_failure)?;
    let mut output_bytes = Vec::new();
    output_bytes
        .try_reserve_exact(encoded_bytes.len())
        .map_err(|error| save_webm_resource_failure(error.to_string()))?;
    output_bytes.extend_from_slice(encoded_bytes);
    let effects = context
        .prepared_effects()
        .map_err(save_webm_effect_failure)?;
    let request = NativeOutputEffectRequest::checked_media(
        NativeOutputNamespace::Output,
        parsed.filename_prefix,
        "webm",
        "video/webm",
        NativeOutputMediaKind::Video,
        0,
        NativeOutputShape::File,
        Arc::from(output_bytes),
        effects.maximum_output_bytes(),
    )
    .map_err(save_webm_effect_failure)?;
    let prepared = effects
        .prepare_output(request, &context.cancellation)
        .map_err(save_webm_effect_failure)?;
    let completion = (|| {
        check_save_webm_cancellation(context)?;
        let outcome = NativeNodeOutcome::Values {
            outputs: vec![NativeValue::Handle {
                value: input_handle,
            }],
            ui: Some(json!({
                "images": [{
                    "transaction_id": prepared.transaction_id(),
                    "batch_index": 0,
                    "type": "output",
                }],
                "animated": [true],
            })),
            effects: vec![prepared.clone()],
        };
        outcome
            .validate()
            .map_err(|error| save_webm_invalid(error.to_string()))?;
        Ok(outcome)
    })();
    match completion {
        Ok(outcome) => Ok(outcome),
        Err(error) => {
            effects
                .rollback_prepared(&prepared)
                .map_err(|rollback| save_webm_rollback_failure(rollback.to_string()))?;
            Err(error)
        }
    }
}

struct SaveWebmInputs<'a> {
    images: &'a NativeOpaqueHandle,
    filename_prefix: &'a str,
    codec: NativeVideoCodec,
    frame_rate: (u64, u64),
    crf: NativeVideoCrf,
    prompt: Option<&'a Value>,
    extra_pnginfo: Option<&'a Value>,
}

fn save_webm_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<SaveWebmInputs<'_>, NativeNodeFailure> {
    if !(5..=7).contains(&inputs.len())
        || inputs.keys().any(|name| {
            !matches!(
                name.as_str(),
                "images"
                    | "filename_prefix"
                    | "codec"
                    | "fps"
                    | "crf"
                    | "prompt"
                    | "extra_pnginfo"
            )
        })
    {
        return Err(save_webm_invalid("received an unexpected input set"));
    }
    let images = interpolation_exact_handle(
        inputs.get("images"),
        NativeHandleKind::Image,
        "IMAGE",
        "images",
    )
    .map_err(|_| save_webm_invalid("images must be an exact IMAGE handle"))?;
    let filename_prefix = match inputs.get("filename_prefix") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) => value.as_str(),
        _ => return Err(save_webm_invalid("filename_prefix must be a string")),
    };
    let codec = match inputs.get("codec") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) if value == "vp9" => NativeVideoCodec::Vp9,
        Some(NativeValue::Primitive {
            value: NativePrimitive::String(value),
        }) if value == "av1" => NativeVideoCodec::Av1,
        _ => return Err(save_webm_invalid("codec must be vp9 or av1")),
    };
    let fps = match inputs.get("fps") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => *value,
        _ => return Err(save_webm_invalid("fps must be a floating-point number")),
    };
    let frame_rate = source_rounded_millisecond_frame_rate(fps)
        .map_err(|error| save_webm_invalid(error.to_string()))?;
    let crf = match inputs.get("crf") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => NativeVideoCrf::checked(*value)
            .map_err(|error| save_webm_invalid(error.to_string()))?,
        _ => return Err(save_webm_invalid("crf must be a floating-point number")),
    };
    let prompt = optional_preserved_json(inputs.get("prompt"), "PROMPT", "prompt")?;
    let extra_pnginfo =
        optional_preserved_json(inputs.get("extra_pnginfo"), "EXTRA_PNGINFO", "extra_pnginfo")?;
    if extra_pnginfo.is_some_and(|value| !value.is_object()) {
        return Err(save_webm_invalid("extra_pnginfo must be a JSON object"));
    }
    Ok(SaveWebmInputs {
        images,
        filename_prefix,
        codec,
        frame_rate,
        crf,
        prompt,
        extra_pnginfo,
    })
}

fn optional_preserved_json<'a>(
    input: Option<&'a NativeValue>,
    type_name: &str,
    name: &str,
) -> Result<Option<&'a Value>, NativeNodeFailure> {
    match input {
        None
        | Some(NativeValue::Primitive {
            value: NativePrimitive::Null,
        })
        | Some(NativeValue::PreservedUnknown {
            value: Value::Null, ..
        }) => Ok(None),
        Some(NativeValue::PreservedUnknown {
            type_name: actual,
            value,
        }) if actual == type_name => Ok(Some(value)),
        _ => Err(save_webm_invalid(format!(
            "{name} must preserve the exact {type_name} source type"
        ))),
    }
}

fn save_webm_metadata(
    prompt: Option<&Value>,
    extra_pnginfo: Option<&Value>,
) -> Result<Vec<(String, String)>, NativeNodeFailure> {
    let extra_entries = extra_pnginfo
        .and_then(Value::as_object)
        .map_or(0, serde_json::Map::len);
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(usize::from(prompt.is_some()).saturating_add(extra_entries))
        .map_err(|error| save_webm_resource_failure(error.to_string()))?;
    if let Some(prompt) = prompt {
        metadata.push((
            "prompt".to_owned(),
            serde_json::to_string(prompt)
                .map_err(|error| save_webm_invalid(error.to_string()))?,
        ));
    }
    if let Some(extra_pnginfo) = extra_pnginfo.and_then(Value::as_object) {
        for (name, value) in extra_pnginfo {
            metadata.push((
                name.clone(),
                serde_json::to_string(value)
                    .map_err(|error| save_webm_invalid(error.to_string()))?,
            ));
        }
    }
    Ok(metadata)
}

fn execute_frame_interpolate(
    context: &NativeNodeContext,
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<NativeNodeOutcome, NativeNodeFailure> {
    check_interpolation_cancellation(context)?;
    let parsed = interpolation_inputs(inputs)?;
    let resolved_model = context
        .handle_store()
        .resolve(
            parsed.interp_model,
            &interpolation_model_type()
                .map_err(|error| interpolation_invalid(error.to_string()))?,
            &context.cancellation,
        )
        .map_err(interpolation_handle_failure)?;
    let resolved_image = context
        .handle_store()
        .resolve(
            parsed.images,
            &image_type().map_err(|error| interpolation_invalid(error.to_string()))?,
            &context.cancellation,
        )
        .map_err(interpolation_handle_failure)?;
    let NativeStoredPayload::Tensor(image_payload) = resolved_image.as_ref() else {
        return Err(interpolation_invalid(
            "IMAGE handle does not contain a tensor payload",
        ));
    };
    if image_payload.role() != NativeTensorRole::Image {
        return Err(interpolation_invalid(
            "IMAGE handle has the wrong tensor role",
        ));
    }
    let image = image_payload
        .image()
        .ok_or_else(|| interpolation_invalid("IMAGE handle has no canonical ImageTensor"))?;
    let (frame_count, _, _, channels) = image.dimensions().map_err(interpolation_tensor_failure)?;

    if frame_count < 2 {
        check_interpolation_cancellation(context)?;
        let outcome = interpolation_outcome(parsed.images.clone());
        outcome
            .validate()
            .map_err(|error| interpolation_invalid(error.to_string()))?;
        drop(resolved_image);
        drop(resolved_model);
        return Ok(outcome);
    }

    if channels != 3 {
        return Err(interpolation_invalid(
            "non-bypass IMAGE input must have three channels",
        ));
    }
    let NativeStoredPayload::Model(stored_model) = resolved_model.as_ref() else {
        return Err(interpolation_invalid(
            "INTERP_MODEL handle does not contain a model payload",
        ));
    };
    let model = stored_model
        .model_payload()
        .frame_interpolation_resource()
        .ok_or_else(|| {
            interpolation_invalid(
                "INTERP_MODEL handle has no concrete frame-interpolation resource",
            )
        })?;
    let compute = context
        .compute_session()
        .map_err(interpolation_compute_failure)?;
    let execution = compute
        .execution_context(context)
        .map_err(interpolation_compute_failure)?;
    if model.stream() != compute.stream()
        || image.tensor().descriptor().stream() != compute.stream()
    {
        return Err(interpolation_invalid(
            "IMAGE, INTERP_MODEL, and compute streams must match",
        ));
    }
    let prepared = cast_to_with_context_exact_native(
        compute.backend(),
        image.tensor(),
        model.dtype(),
        DeviceId::CPU,
        false,
        false,
        &execution,
    )
    .map_err(|error| interpolation_operator_failure(context, error))?;
    let output = model
        .interpolate_sequence(compute.backend(), &prepared, parsed.multiplier, &execution)
        .map_err(|error| interpolation_model_failure(context, error))?;
    let output = cast_to_with_context_exact_native(
        compute.backend(),
        &output,
        DType::F32,
        DeviceId::CPU,
        false,
        false,
        &execution,
    )
    .map_err(|error| interpolation_operator_failure(context, error))?;
    let output = ImageTensor::from_tensor(output).map_err(interpolation_tensor_failure)?;
    let payload = NativeTensorPayload::from_image(NativeTensorRole::Image, output)
        .map_err(interpolation_payload_failure)?;
    drop(resolved_image);
    drop(resolved_model);
    check_interpolation_cancellation(context)?;
    let output_handle = context
        .handle_store()
        .publish(
            NativeStoredPayload::Tensor(Arc::new(payload)),
            &context.cancellation,
        )
        .map_err(interpolation_handle_failure)?;
    let completion = (|| {
        check_interpolation_cancellation(context)?;
        let outcome = interpolation_outcome(output_handle.clone());
        outcome
            .validate()
            .map_err(|error| interpolation_invalid(error.to_string()))?;
        Ok(outcome)
    })();
    match completion {
        Ok(outcome) => Ok(outcome),
        Err(error) => {
            rollback_interpolation_output(context, &output_handle)?;
            Err(error)
        }
    }
}

struct InterpolationInputs<'a> {
    interp_model: &'a NativeOpaqueHandle,
    images: &'a NativeOpaqueHandle,
    multiplier: u64,
}

fn interpolation_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<InterpolationInputs<'_>, NativeNodeFailure> {
    if inputs.len() != 3
        || inputs
            .keys()
            .any(|name| !matches!(name.as_str(), "interp_model" | "images" | "multiplier"))
    {
        return Err(interpolation_invalid(
            "requires exactly interp_model, images, and multiplier",
        ));
    }
    let interp_model = interpolation_exact_handle(
        inputs.get("interp_model"),
        NativeHandleKind::Model,
        "INTERP_MODEL",
        "interp_model",
    )?;
    let images = interpolation_exact_handle(
        inputs.get("images"),
        NativeHandleKind::Image,
        "IMAGE",
        "images",
    )?;
    let multiplier = match inputs.get("multiplier") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) if (2..=16).contains(value) => u64::try_from(*value)
            .map_err(|_| interpolation_invalid("multiplier must be between 2 and 16"))?,
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) if (2..=16).contains(value) => *value,
        _ => {
            return Err(interpolation_invalid(
                "multiplier must be an integer between 2 and 16",
            ));
        }
    };
    Ok(InterpolationInputs {
        interp_model,
        images,
        multiplier,
    })
}

fn interpolation_exact_handle<'a>(
    value: Option<&'a NativeValue>,
    kind: NativeHandleKind,
    type_id: &str,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(interpolation_invalid(format!("{name} must be a handle")));
    };
    if value.handle_type().kind != kind || value.handle_type().type_id != type_id {
        return Err(interpolation_invalid(format!(
            "{name} must be an exact {type_id} handle"
        )));
    }
    Ok(value)
}

fn interpolation_outcome(handle: NativeOpaqueHandle) -> NativeNodeOutcome {
    NativeNodeOutcome::Values {
        outputs: vec![NativeValue::Handle { value: handle }],
        ui: None,
        effects: Vec::new(),
    }
}

fn rollback_interpolation_output(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<(), NativeNodeFailure> {
    context
        .handle_store()
        .revoke(handle, &CancellationToken::default())
        .map_err(|error| NativeNodeFailure {
            code: "frame_interpolation_rollback_failed".to_owned(),
            message: format!("FrameInterpolate could not revoke partial output: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        })
}

fn check_interpolation_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interpolation_interrupted())
}

fn interpolation_handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interpolation_interrupted()
    } else {
        NativeNodeFailure {
            code: "invalid_frame_interpolation_handle".to_owned(),
            message: format!("FrameInterpolate input handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn interpolation_compute_failure(error: NativeNodeContractError) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "native_compute_unavailable".to_owned(),
        message: format!("FrameInterpolate compute session is unavailable: {error}"),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn interpolation_tensor_failure(error: TensorError) -> NativeNodeFailure {
    match error {
        TensorError::Cancelled => interpolation_interrupted(),
        error @ (TensorError::AllocationFailed { .. }
        | TensorError::ResourceLimitExceeded { .. }
        | TensorError::WorkspaceAuthorizationExceeded { .. }) => NativeNodeFailure {
            code: "frame_interpolation_resource_exhausted".to_owned(),
            message: format!("FrameInterpolate exhausted a bounded resource: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
        error => interpolation_invalid(error.to_string()),
    }
}

fn interpolation_operator_failure(
    context: &NativeNodeContext,
    error: OperatorIndirectionError,
) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        return interpolation_interrupted();
    }
    match error {
        OperatorIndirectionError::Cancelled => interpolation_interrupted(),
        OperatorIndirectionError::Tensor(error) => interpolation_tensor_failure(error),
        error => interpolation_invalid(error.to_string()),
    }
}

fn interpolation_payload_failure(error: NativeTensorPayloadError) -> NativeNodeFailure {
    match error {
        NativeTensorPayloadError::Tensor(error) => interpolation_tensor_failure(error),
        error => interpolation_invalid(error.to_string()),
    }
}

fn interpolation_model_failure(
    context: &NativeNodeContext,
    error: FrameInterpolationError,
) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        return interpolation_interrupted();
    }
    match error {
        FrameInterpolationError::Cancelled => interpolation_interrupted(),
        FrameInterpolationError::ResourceExhausted(message) => NativeNodeFailure {
            code: "frame_interpolation_resource_exhausted".to_owned(),
            message: format!("FrameInterpolate exhausted a bounded resource: {message}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
        FrameInterpolationError::Tensor(error) => interpolation_tensor_failure(error),
        error => NativeNodeFailure {
            code: "execution_error".to_owned(),
            message: format!("FrameInterpolate execution failed: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
    }
}

fn interpolation_interrupted() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "FrameInterpolate execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

fn interpolation_invalid(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: format!("FrameInterpolate: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn components_input(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    if inputs.len() != 1 || !inputs.contains_key("video") {
        return Err(components_invalid(
            "GetVideoComponents requires exactly one video input",
        ));
    }
    exact_components_handle(inputs.get("video"))
}

fn exact_components_handle(
    value: Option<&NativeValue>,
) -> Result<&NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(components_invalid(
            "GetVideoComponents video must be a handle",
        ));
    };
    if value.handle_type().kind != NativeHandleKind::Video || value.handle_type().type_id != "VIDEO"
    {
        return Err(components_invalid(
            "GetVideoComponents video must be an exact VIDEO handle",
        ));
    }
    Ok(value)
}

fn project_video_frames(
    context: &NativeNodeContext,
    frames: &sim_tensor::Tensor,
) -> Result<ImageTensor, NativeNodeFailure> {
    let descriptor = frames.descriptor();
    if descriptor.device() != DeviceId::CPU {
        return Err(components_invalid("VIDEO frames must use CPU storage"));
    }
    let [_, _, _, channels] = descriptor.shape() else {
        return Err(components_invalid("VIDEO frames must use BHWC rank four"));
    };
    if !matches!(channels, 1 | 3 | 4) {
        return Err(components_invalid(
            "VIDEO frames must have one, three, or four channels",
        ));
    }
    let contiguous = descriptor
        .is_contiguous()
        .map_err(components_tensor_failure)?;
    match (descriptor.dtype(), contiguous) {
        (DType::F32, true) => {
            ImageTensor::from_tensor(frames.clone()).map_err(components_tensor_failure)
        }
        (DType::F32, false) => {
            let compute = context.compute_session().map_err(|error| {
                components_invalid(format!(
                    "VIDEO frame conversion needs compute access: {error}"
                ))
            })?;
            let execution = compute
                .execution_context(context)
                .map_err(|error| components_invalid(error.to_string()))?;
            let frames = contiguous_with_context_exact_native(
                compute.backend(),
                frames,
                MemoryFormatReference::Layout(Layout::Contiguous),
                &execution,
            )
            .map_err(|error| components_operation_failure(context, error.to_string()))?;
            ImageTensor::from_tensor(frames).map_err(components_tensor_failure)
        }
        (DType::U8, _) => {
            let compute = context.compute_session().map_err(|error| {
                components_invalid(format!(
                    "VIDEO frame conversion needs compute access: {error}"
                ))
            })?;
            let execution = compute
                .execution_context(context)
                .map_err(|error| components_invalid(error.to_string()))?;
            let frames = cast_to_with_context_exact_native(
                compute.backend(),
                frames,
                DType::F32,
                DeviceId::CPU,
                false,
                false,
                &execution,
            )
            .map_err(|error| components_operation_failure(context, error.to_string()))?;
            let frames = div_with_context_exact_native(
                compute.backend(),
                &frames,
                ElementwiseOperand::Scalar(Scalar::Float(255.0)),
                &execution,
            )
            .map_err(|error| components_operation_failure(context, error.to_string()))?;
            check_components_cancellation(context)?;
            ImageTensor::from_tensor(frames).map_err(components_tensor_failure)
        }
        _ => Err(components_invalid(
            "VIDEO frames must use F32 or U8 storage",
        )),
    }
}

fn components_operation_failure(context: &NativeNodeContext, message: String) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        return NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: "GetVideoComponents execution was interrupted".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        };
    }
    components_invalid(message)
}

fn publish_component(
    context: &NativeNodeContext,
    payload: NativeStoredPayload,
    published: &mut Vec<NativeOpaqueHandle>,
) -> Result<NativeOpaqueHandle, NativeNodeFailure> {
    check_components_cancellation(context)?;
    let handle = context
        .handle_store()
        .publish(payload, &context.cancellation)
        .map_err(components_handle_failure)?;
    published.push(handle.clone());
    Ok(handle)
}

fn rollback_components(
    context: &NativeNodeContext,
    published: &[NativeOpaqueHandle],
) -> Result<(), NativeNodeFailure> {
    let cleanup = CancellationToken::default();
    for handle in published.iter().rev() {
        context
            .handle_store()
            .revoke(handle, &cleanup)
            .map_err(|error| NativeNodeFailure {
                code: "video_component_rollback_failed".to_owned(),
                message: format!("GetVideoComponents could not revoke partial output: {error}"),
                kind: NativeNodeFailureKind::Failure,
                retryable: false,
            })?;
    }
    Ok(())
}

fn check_components_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context.cancellation.check().map_err(|_| NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "GetVideoComponents execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    })
}

fn components_handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        return NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: "GetVideoComponents execution was interrupted".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        };
    }
    components_invalid(format!("video component handle is unavailable: {error}"))
}

fn components_tensor_failure(error: TensorError) -> NativeNodeFailure {
    if matches!(error, TensorError::Cancelled) {
        return NativeNodeFailure {
            code: "execution_interrupted".to_owned(),
            message: "GetVideoComponents execution was interrupted".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: false,
        };
    }
    components_invalid(error.to_string())
}

fn components_invalid(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_video_components".to_owned(),
        message: format!("GetVideoComponents: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

struct ParsedInputs<'a> {
    images: &'a NativeOpaqueHandle,
    audio: Option<&'a NativeOpaqueHandle>,
    frame_rate: (u64, u64),
    bit_depth: NativeVideoBitDepth,
}

fn parse_inputs(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<ParsedInputs<'_>, NativeNodeFailure> {
    if !(2..=4).contains(&inputs.len())
        || inputs
            .keys()
            .any(|name| !matches!(name.as_str(), "images" | "fps" | "audio" | "bit_depth"))
    {
        return Err(invalid_inputs(
            "CreateVideo received an unexpected input set",
        ));
    }
    let images = exact_handle(
        inputs.get("images"),
        NativeHandleKind::Image,
        "IMAGE",
        "images",
    )?;
    let audio = inputs
        .get("audio")
        .map(|value| exact_handle(Some(value), NativeHandleKind::Audio, "AUDIO", "audio"))
        .transpose()?;
    let fps = match inputs.get("fps") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => *value,
        _ => {
            return Err(invalid_inputs(
                "CreateVideo fps must be a floating-point number",
            ));
        }
    };
    let frame_rate = exact_positive_f64_fraction(fps)
        .filter(|(numerator, denominator)| {
            let value = *numerator as f64 / *denominator as f64;
            (1.0..=120.0).contains(&value)
        })
        .ok_or_else(|| invalid_inputs("CreateVideo fps must be finite and in 1 through 120"))?;
    let bit_depth = match inputs.get("bit_depth") {
        None => NativeVideoBitDepth::Eight,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(8) | NativePrimitive::UnsignedInteger(8),
        }) => NativeVideoBitDepth::Eight,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(10) | NativePrimitive::UnsignedInteger(10),
        }) => NativeVideoBitDepth::Ten,
        _ => return Err(invalid_inputs("CreateVideo bit_depth must be 8 or 10")),
    };
    Ok(ParsedInputs {
        images,
        audio,
        frame_rate,
        bit_depth,
    })
}

fn exact_handle<'a>(
    value: Option<&'a NativeValue>,
    kind: NativeHandleKind,
    type_id: &str,
    name: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    let Some(NativeValue::Handle { value }) = value else {
        return Err(invalid_inputs(format!(
            "CreateVideo {name} must be a handle"
        )));
    };
    if value.handle_type().kind != kind || value.handle_type().type_id != type_id {
        return Err(invalid_inputs(format!(
            "CreateVideo {name} must be an exact {type_id} handle"
        )));
    }
    Ok(value)
}

fn exact_positive_f64_fraction(value: f64) -> Option<(u64, u64)> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    if bits >> 63 != 0 {
        return None;
    }
    let exponent_bits = i32::try_from((bits >> 52) & 0x7ff).ok()?;
    if exponent_bits == 0 || exponent_bits == 0x7ff {
        return None;
    }
    let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    let binary_exponent = exponent_bits - 1023 - 52;
    let (numerator, denominator) = if binary_exponent >= 0 {
        (
            significand.checked_shl(u32::try_from(binary_exponent).ok()?)?,
            1,
        )
    } else {
        (
            significand,
            1_u64.checked_shl(u32::try_from(-binary_exponent).ok()?)?,
        )
    };
    let divisor = greatest_common_divisor(numerator, denominator);
    Some((numerator / divisor, denominator / divisor))
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn check_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| interrupted_failure())
}

fn handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        interrupted_failure()
    } else {
        NativeNodeFailure {
            code: "invalid_media_handle".to_owned(),
            message: format!("CreateVideo input handle is unavailable: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        }
    }
}

fn invalid_payload(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_video_components".to_owned(),
        message: format!("CreateVideo: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn invalid_inputs(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_node_inputs".to_owned(),
        message: message.into(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn interrupted_failure() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "CreateVideo execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

fn check_save_webm_cancellation(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|_| save_webm_interrupted())
}

fn save_webm_handle_failure(error: NativeHandleStoreError) -> NativeNodeFailure {
    if matches!(error, NativeHandleStoreError::Cancelled) {
        save_webm_interrupted()
    } else {
        save_webm_invalid(format!("IMAGE handle is unavailable: {error}"))
    }
}

fn save_webm_encode_failure(error: NativeWebmEncodeServiceError) -> NativeNodeFailure {
    match error {
        NativeWebmEncodeServiceError::Cancelled => save_webm_interrupted(),
        NativeWebmEncodeServiceError::Busy => NativeNodeFailure {
            code: "webm_encoder_busy".to_owned(),
            message: "SaveWEBM encoder is busy".to_owned(),
            kind: NativeNodeFailureKind::Failure,
            retryable: true,
        },
        NativeWebmEncodeServiceError::Unavailable => {
            save_webm_service_failure("WebM encoding service is unavailable")
        }
        NativeWebmEncodeServiceError::ResourceExhausted => {
            save_webm_resource_failure("WebM encoding exhausted its reviewed resources")
        }
        NativeWebmEncodeServiceError::InvalidRequest
        | NativeWebmEncodeServiceError::InvalidProjection => {
            save_webm_invalid(error.to_string())
        }
        NativeWebmEncodeServiceError::Execution(message) => NativeNodeFailure {
            code: "webm_encode_failed".to_owned(),
            message: format!("SaveWEBM encoding failed: {message}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
    }
}

fn save_webm_effect_failure(error: NativeEffectServiceError) -> NativeNodeFailure {
    match error {
        NativeEffectServiceError::Cancelled => save_webm_interrupted(),
        NativeEffectServiceError::Unavailable => {
            save_webm_service_failure("prepared output service is unavailable")
        }
        error => NativeNodeFailure {
            code: "webm_output_prepare_failed".to_owned(),
            message: format!("SaveWEBM could not prepare its output: {error}"),
            kind: NativeNodeFailureKind::Failure,
            retryable: false,
        },
    }
}

fn save_webm_tensor_failure(error: TensorError) -> NativeNodeFailure {
    match error {
        TensorError::Cancelled => save_webm_interrupted(),
        TensorError::AllocationFailed { .. }
        | TensorError::ResourceLimitExceeded { .. }
        | TensorError::WorkspaceAuthorizationExceeded { .. } => {
            save_webm_resource_failure(error.to_string())
        }
        error => save_webm_invalid(error.to_string()),
    }
}

fn save_webm_service_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "webm_service_unavailable".to_owned(),
        message: format!("SaveWEBM: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: true,
    }
}

fn save_webm_resource_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "webm_resource_exhausted".to_owned(),
        message: format!("SaveWEBM exhausted a bounded resource: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn save_webm_rollback_failure(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "webm_output_rollback_failed".to_owned(),
        message: format!("SaveWEBM could not roll back prepared output: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

fn save_webm_interrupted() -> NativeNodeFailure {
    NativeNodeFailure {
        code: "execution_interrupted".to_owned(),
        message: "SaveWEBM execution was interrupted".to_owned(),
        kind: NativeNodeFailureKind::Interrupted,
        retryable: false,
    }
}

fn save_webm_invalid(message: impl Into<String>) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_save_webm".to_owned(),
        message: format!("SaveWEBM: {}", message.into()),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeEncodedWebm, NativeHandleStore, NativeHandleStoreIdentity, NativeNodeComputeSession,
        NativeNodeServiceIdentity, NativeNodeServices, NativePreparedEffectKind,
        NativePreparedEffectRequest, NativePreparedEffectService, NativeResolvedPayload,
        NativeResolvedPayloadRetention, NativeStoredModelPayload, NativeWebmEncodeService,
        NativeWebmEncodeServiceIdentity,
    };
    use sim_media::{NativeAudioPayload, NativeVideoPixelFormat};
    use sim_model::{NativeFrameInterpolationModel, NativeModelPayload};
    use sim_tensor::{
        CpuWorkspaceAuthority, DType, DeviceId, ImageTensor, NativeTensorPayload, StreamId,
        TensorDescriptor,
    };
    use sim_types::{AttemptId, CancellationToken, NodeId, PromptId};
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::{
        error::Error,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use uuid::Uuid;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/video-comfy-node-0124/fixture.json"
    ));
    const COMPONENTS_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/video-comfy-node-0207/fixture.json"
    ));
    const INTERPOLATION_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/video-comfy-node-0190/fixture.json"
    ));
    const SAVE_WEBM_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/video-comfy-node-0602/fixture.json"
    ));

    #[derive(Debug)]
    struct TestWebmService {
        identity: NativeWebmEncodeServiceIdentity,
        backend: Arc<sim_tensor::CpuBackend>,
        requests: Mutex<Vec<(NativeVideoCodec, (u64, u64), u64, Vec<(String, String)>)>>,
    }

    impl NativeWebmEncodeService for TestWebmService {
        fn identity(&self) -> &NativeWebmEncodeServiceIdentity {
            &self.identity
        }

        fn encode_webm(
            &self,
            request: NativeWebmEncodeRequest,
            context: &sim_tensor::ExecutionContext<'_>,
        ) -> BoxFuture<'static, Result<NativeEncodedWebm, NativeWebmEncodeServiceError>> {
            let (images, codec, frame_rate, crf, metadata) = request.into_parts();
            let result = (|| {
                context
                    .check()
                    .map_err(|_| NativeWebmEncodeServiceError::Cancelled)?;
                self.requests
                    .lock()
                    .map_err(|_| {
                        NativeWebmEncodeServiceError::Execution(
                            "test request lock was poisoned".to_owned(),
                        )
                    })?
                    .push((codec, frame_rate, crf.bits(), metadata));
                let (frame_count, height, width, channels) = images
                    .dimensions()
                    .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                let encoded = match codec {
                    NativeVideoCodec::Vp9 => b"HPPPT".as_slice(),
                    NativeVideoCodec::Av1 => b"HA1T".as_slice(),
                    NativeVideoCodec::H264 => {
                        return Err(NativeWebmEncodeServiceError::InvalidRequest);
                    }
                };
                let descriptor = TensorDescriptor::contiguous(
                    vec![u64::try_from(encoded.len()).map_err(|_| {
                        NativeWebmEncodeServiceError::ResourceExhausted
                    })?],
                    DType::U8,
                    DeviceId::CPU,
                    context.stream,
                )
                .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                let (bytes, _) = self
                    .backend
                    .upload_bytes(descriptor, encoded, context)
                    .map_err(|error| NativeWebmEncodeServiceError::Execution(error.to_string()))?;
                NativeEncodedWebm::checked(
                    bytes,
                    Sha256::digest(encoded).into(),
                    codec,
                    (width, height),
                    frame_rate,
                    frame_count,
                    match (codec, channels) {
                        (NativeVideoCodec::Vp9, 4) => NativeVideoPixelFormat::Yuva420p,
                        (NativeVideoCodec::Vp9, _) => NativeVideoPixelFormat::Yuv420p,
                        (NativeVideoCodec::Av1, _) => NativeVideoPixelFormat::Yuv420p10le,
                        (NativeVideoCodec::H264, _) => {
                            return Err(NativeWebmEncodeServiceError::InvalidRequest);
                        }
                    },
                    if codec == NativeVideoCodec::Av1 {
                        NativeVideoBitDepth::Ten
                    } else {
                        NativeVideoBitDepth::Eight
                    },
                    codec == NativeVideoCodec::Vp9 && channels == 4,
                )
            })();
            Box::pin(async move { result })
        }
    }

    #[derive(Debug)]
    struct TestSaveWebmEffects {
        identity: NativeNodeServiceIdentity,
        requests: Mutex<Vec<NativeOutputEffectRequest>>,
        prepared: Mutex<Vec<NativePreparedEffectRequest>>,
        cancel_after_prepare: Mutex<Option<CancellationToken>>,
    }

    impl NativePreparedEffectService for TestSaveWebmEffects {
        fn identity(&self) -> &NativeNodeServiceIdentity {
            &self.identity
        }

        fn maximum_output_bytes(&self) -> u64 {
            1024 * 1024
        }

        fn prepare_output(
            &self,
            request: NativeOutputEffectRequest,
            cancellation: &CancellationToken,
        ) -> Result<NativePreparedEffectRequest, NativeEffectServiceError> {
            cancellation
                .check()
                .map_err(|_| NativeEffectServiceError::Cancelled)?;
            let mut requests = self
                .requests
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            requests.push(request.clone());
            let ordinal = u128::try_from(requests.len())
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            drop(requests);
            let prepared = NativePreparedEffectRequest::checked(
                self.identity.service_id(),
                Uuid::from_u128(0x60200 + ordinal),
                NativePreparedEffectKind::Output,
                request.request_digest_sha256(),
            )
            .map_err(|_| NativeEffectServiceError::Rejected)?;
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .push(prepared.clone());
            if let Some(cancellation) = self
                .cancel_after_prepare
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .take()
            {
                cancellation.cancel();
            }
            Ok(prepared)
        }

        fn rollback_prepared(
            &self,
            request: &NativePreparedEffectRequest,
        ) -> Result<(), NativeEffectServiceError> {
            let mut prepared = self
                .prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?;
            let index = prepared
                .iter()
                .position(|candidate| candidate == request)
                .ok_or(NativeEffectServiceError::InvalidTicket)?;
            prepared.remove(index);
            Ok(())
        }

        fn rollback_all_prepared(&self) -> Result<(), NativeEffectServiceError> {
            self.prepared
                .lock()
                .map_err(|_| NativeEffectServiceError::Rejected)?
                .clear();
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TestRetention;

    impl NativeResolvedPayloadRetention for TestRetention {}

    #[derive(Debug)]
    struct TestStore {
        identity: NativeHandleStoreIdentity,
        attempt_id: AttemptId,
        next_identifier: AtomicU64,
        cancel_after_publish: std::sync::atomic::AtomicBool,
        values: Mutex<BTreeMap<String, Arc<NativeStoredPayload>>>,
    }

    impl TestStore {
        fn new(attempt_id: AttemptId) -> Result<Arc<Self>, Box<dyn Error>> {
            Ok(Arc::new(Self {
                identity: NativeHandleStoreIdentity::new(
                    Uuid::from_u128(0x12401),
                    Uuid::from_u128(0x12402),
                )?,
                attempt_id,
                next_identifier: AtomicU64::new(1),
                cancel_after_publish: std::sync::atomic::AtomicBool::new(false),
                values: Mutex::new(BTreeMap::new()),
            }))
        }

        fn cancel_after_next_publish(&self) {
            self.cancel_after_publish.store(true, Ordering::Release);
        }

        fn count(&self) -> Result<usize, NativeHandleStoreError> {
            self.values.lock().map(|values| values.len()).map_err(|_| {
                NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
            })
        }

        fn payload(
            &self,
            handle: &NativeOpaqueHandle,
        ) -> Result<Arc<NativeStoredPayload>, NativeHandleStoreError> {
            self.values
                .lock()
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))
        }
    }

    impl NativeHandleStore for TestStore {
        fn identity(&self) -> NativeHandleStoreIdentity {
            self.identity
        }

        fn attempt_id(&self) -> AttemptId {
            self.attempt_id
        }

        fn resolve(
            &self,
            handle: &NativeOpaqueHandle,
            expected_type: &NativeHandleType,
            cancellation: &CancellationToken,
        ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            if handle.store_identity() != self.identity {
                return Err(NativeHandleStoreError::WrongStore);
            }
            if handle.handle_type() != expected_type {
                return Err(NativeHandleStoreError::WrongType {
                    expected: expected_type.type_id.clone(),
                    actual: handle.handle_type().type_id.clone(),
                });
            }
            let payload = self
                .values
                .lock()
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .get(handle.identifier())
                .cloned()
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            if handle.digest_sha256() != Some(payload.digest_sha256().as_str()) {
                return Err(NativeHandleStoreError::DigestMismatch);
            }
            NativeResolvedPayload::checked(payload, Arc::new(TestRetention)).map_err(Into::into)
        }

        fn publish(
            &self,
            payload: NativeStoredPayload,
            cancellation: &CancellationToken,
        ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            payload.validate()?;
            let handle_type = payload.handle_type()?;
            let digest = payload.digest_sha256();
            let identifier = format!(
                "video-component-{}",
                self.next_identifier.fetch_add(1, Ordering::AcqRel)
            );
            let handle = NativeOpaqueHandle::new(
                handle_type,
                self.identity,
                identifier.clone(),
                1,
                Some(digest),
            )?;
            self.values
                .lock()
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .insert(identifier, Arc::new(payload));
            if self.cancel_after_publish.swap(false, Ordering::AcqRel) {
                cancellation.cancel();
            }
            Ok(handle)
        }

        fn revoke(
            &self,
            handle: &NativeOpaqueHandle,
            cancellation: &CancellationToken,
        ) -> Result<(), NativeHandleStoreError> {
            cancellation
                .check()
                .map_err(|_| NativeHandleStoreError::Cancelled)?;
            self.values
                .lock()
                .map_err(|_| {
                    NativeHandleStoreError::Rejected("test store lock was poisoned".to_owned())
                })?
                .remove(handle.identifier())
                .ok_or_else(|| NativeHandleStoreError::Missing(handle.identifier().to_owned()))?;
            Ok(())
        }
    }

    struct Harness {
        store: Arc<TestStore>,
        backend: Arc<sim_tensor::CpuBackend>,
        workspace: CpuWorkspaceAuthority,
        attempt_id: AttemptId,
        node_id: NodeId,
        scratch_bytes: u64,
    }

    impl Harness {
        fn new() -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x12403));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
            Ok(Self {
                store: TestStore::new(attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("create-video-test".to_owned()),
                scratch_bytes: 1024 * 1024,
            })
        }

        fn interpolation() -> Result<Self, Box<dyn Error>> {
            let attempt_id = AttemptId(Uuid::from_u128(0x19003));
            let (backend, workspace) = CpuWorkspaceAuthority::create_backend(512 * 1024 * 1024)?;
            Ok(Self {
                store: TestStore::new(attempt_id)?,
                backend: Arc::new(backend),
                workspace,
                attempt_id,
                node_id: NodeId("frame-interpolate-test".to_owned()),
                scratch_bytes: 256 * 1024 * 1024,
            })
        }

        fn image_handle(
            &self,
        ) -> Result<(NativeOpaqueHandle, sim_tensor::StorageId), Box<dyn Error>> {
            self.image_handle_with_shape(2, 2, 2, 3, &[0.25; 24])
        }

        fn image_handle_with_shape(
            &self,
            batch: u64,
            height: u64,
            width: u64,
            channels: u64,
            values: &[f32],
        ) -> Result<(NativeOpaqueHandle, sim_tensor::StorageId), Box<dyn Error>> {
            let cancellation = CancellationToken::default();
            let context = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(0)?,
                &cancellation,
            );
            let image = ImageTensor::from_f32(
                &self.backend,
                &context,
                batch,
                height,
                width,
                channels,
                values,
            )?;
            let storage_id = image.tensor().storage_id();
            let handle = self.store.publish(
                NativeStoredPayload::Tensor(Arc::new(NativeTensorPayload::from_image(
                    NativeTensorRole::Image,
                    image,
                )?)),
                &cancellation,
            )?;
            Ok((handle, storage_id))
        }

        fn interpolation_model_handle(
            &self,
        ) -> Result<(NativeOpaqueHandle, Arc<NativeFrameInterpolationModel>), Box<dyn Error>>
        {
            let cancellation = CancellationToken::default();
            let context = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(self.scratch_bytes)?,
                &cancellation,
            );
            let model = Arc::new(NativeFrameInterpolationModel::reduced_rife_test_fixture(
                &self.backend,
                &context,
            )?);
            let payload = Arc::new(NativeModelPayload::frame_interpolation(model.clone())?);
            let stored = NativeStoredModelPayload::model_resource(payload)?;
            let handle = self
                .store
                .publish(NativeStoredPayload::Model(Arc::new(stored)), &cancellation)?;
            Ok((handle, model))
        }

        fn audio_handle(
            &self,
        ) -> Result<(NativeOpaqueHandle, sim_tensor::StorageId), Box<dyn Error>> {
            let descriptor = TensorDescriptor::contiguous(
                vec![1, 1, 4],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?;
            let cancellation = CancellationToken::default();
            let context = self.backend.execution_context(
                StreamId::DEFAULT,
                self.workspace.authorize_workspace(0)?,
                &cancellation,
            );
            let (waveform, _) = self.backend.upload_bytes(descriptor, &[0; 16], &context)?;
            let storage_id = waveform.storage_id();
            let handle = self.store.publish(
                NativeStoredPayload::Audio(Arc::new(NativeAudioPayload::checked(
                    waveform, 48_000,
                )?)),
                &cancellation,
            )?;
            Ok((handle, storage_id))
        }

        fn video_handle(
            &self,
            frames: sim_tensor::Tensor,
            audio: Option<NativeAudioPayload>,
            frame_rate: (u64, u64),
            bit_depth: NativeVideoBitDepth,
        ) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
            Ok(self.store.publish(
                NativeStoredPayload::Video(Arc::new(NativeVideoPayload::checked(
                    frames,
                    frame_rate.0,
                    frame_rate.1,
                    bit_depth,
                    audio,
                    None,
                    BTreeMap::new(),
                )?)),
                &CancellationToken::default(),
            )?)
        }

        fn context(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            self.context_with_scratch(cancellation, self.scratch_bytes)
        }

        fn context_with_scratch(
            &self,
            cancellation: CancellationToken,
            scratch_bytes: u64,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            let scratch = self.workspace.authorize_workspace(scratch_bytes)?;
            let compute = NativeNodeComputeSession::checked(
                NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x12405),
                    self.attempt_id,
                    self.node_id.clone(),
                )?,
                self.backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            Ok(NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x12404)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                NativeNodeServices::checked(None, None, Some(compute))?,
            )?)
        }

        fn context_without_compute(
            &self,
            cancellation: CancellationToken,
        ) -> Result<NativeNodeContext, Box<dyn Error>> {
            Ok(NativeNodeContext::new(
                PromptId(Uuid::from_u128(0x19004)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                self.workspace.authorize_workspace(0)?,
                self.store.clone(),
            )?)
        }

        fn save_webm_context(
            &self,
            cancellation: CancellationToken,
            cancel_after_prepare: bool,
        ) -> Result<
            (
                NativeNodeContext,
                Arc<TestWebmService>,
                Arc<TestSaveWebmEffects>,
            ),
            Box<dyn Error>,
        > {
            let scratch = self.workspace.authorize_workspace(self.scratch_bytes)?;
            let compute = NativeNodeComputeSession::checked(
                NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x60205),
                    self.attempt_id,
                    self.node_id.clone(),
                )?,
                self.backend.clone(),
                StreamId::DEFAULT,
                &scratch,
            )?;
            let webm = Arc::new(TestWebmService {
                identity: NativeWebmEncodeServiceIdentity::checked("6".repeat(64))?,
                backend: self.backend.clone(),
                requests: Mutex::new(Vec::new()),
            });
            let effects = Arc::new(TestSaveWebmEffects {
                identity: NativeNodeServiceIdentity::checked(
                    Uuid::from_u128(0x60206),
                    self.attempt_id,
                    self.node_id.clone(),
                )?,
                requests: Mutex::new(Vec::new()),
                prepared: Mutex::new(Vec::new()),
                cancel_after_prepare: Mutex::new(
                    cancel_after_prepare.then_some(cancellation.clone()),
                ),
            });
            let services = NativeNodeServices::checked(None, Some(effects.clone()), Some(compute))?
                .with_webm_encode(webm.clone())?;
            let context = NativeNodeContext::new_with_services(
                PromptId(Uuid::from_u128(0x60204)),
                self.attempt_id,
                self.node_id.clone(),
                cancellation,
                scratch,
                self.store.clone(),
                services,
            )?;
            Ok((context, webm, effects))
        }

        fn inputs(
            &self,
            image: NativeOpaqueHandle,
            fps: f64,
            audio: Option<NativeOpaqueHandle>,
            bit_depth: Option<NativePrimitive>,
        ) -> BTreeMap<String, NativeValue> {
            let mut inputs = BTreeMap::from([
                ("images".to_owned(), NativeValue::Handle { value: image }),
                (
                    "fps".to_owned(),
                    NativeValue::Primitive {
                        value: NativePrimitive::Number(fps),
                    },
                ),
            ]);
            if let Some(audio) = audio {
                inputs.insert("audio".to_owned(), NativeValue::Handle { value: audio });
            }
            if let Some(bit_depth) = bit_depth {
                inputs.insert(
                    "bit_depth".to_owned(),
                    NativeValue::Primitive { value: bit_depth },
                );
            }
            inputs
        }

        fn video(&self, outcome: NativeNodeOutcome) -> Result<NativeVideoPayload, Box<dyn Error>> {
            let NativeNodeOutcome::Values {
                outputs,
                ui,
                effects,
            } = outcome
            else {
                return Err("CreateVideo did not return values".into());
            };
            assert!(ui.is_none());
            assert!(effects.is_empty());
            let Some(NativeValue::Handle { value }) = outputs.first() else {
                return Err("CreateVideo output handle is absent".into());
            };
            let resolved =
                self.store
                    .resolve(value, &video_type()?, &CancellationToken::default())?;
            let NativeStoredPayload::Video(video) = resolved.as_ref() else {
                return Err("CreateVideo output is not a VIDEO payload".into());
            };
            Ok(video.as_ref().clone())
        }
    }

    fn executable() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == CLASS_TYPE => Some(node),
                _ => None,
            })
            .ok_or_else(|| "CreateVideo executable binding is absent".into())
    }

    fn components_executable() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == COMPONENTS_CLASS_TYPE => Some(node),
                _ => None,
            })
            .ok_or_else(|| "GetVideoComponents executable binding is absent".into())
    }

    fn interpolation_executable() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == INTERPOLATE_CLASS_TYPE => Some(node),
                _ => None,
            })
            .ok_or_else(|| "FrameInterpolate executable binding is absent".into())
    }

    fn save_webm_executable() -> Result<Arc<dyn NativeNode>, Box<dyn Error>> {
        native_node_bindings()?
            .into_iter()
            .find_map(|binding| match binding {
                NativeNodeBinding::Executable {
                    descriptor, node, ..
                } if descriptor.class_type == SAVE_WEBM_CLASS_TYPE => Some(node),
                _ => None,
            })
            .ok_or_else(|| "SaveWEBM executable binding is absent".into())
    }

    fn save_webm_test_inputs(
        images: NativeOpaqueHandle,
        codec: &str,
        fps: f64,
        crf: f64,
        prompt: Option<Value>,
        extra_pnginfo: Option<Value>,
    ) -> BTreeMap<String, NativeValue> {
        let mut inputs = BTreeMap::from([
            ("images".to_owned(), NativeValue::Handle { value: images }),
            (
                "filename_prefix".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String("video/ComfyUI".to_owned()),
                },
            ),
            (
                "codec".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::String(codec.to_owned()),
                },
            ),
            (
                "fps".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(fps),
                },
            ),
            (
                "crf".to_owned(),
                NativeValue::Primitive {
                    value: NativePrimitive::Number(crf),
                },
            ),
        ]);
        if let Some(prompt) = prompt {
            inputs.insert(
                "prompt".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "PROMPT".to_owned(),
                    value: prompt,
                },
            );
        }
        if let Some(extra_pnginfo) = extra_pnginfo {
            inputs.insert(
                "extra_pnginfo".to_owned(),
                NativeValue::PreservedUnknown {
                    type_name: "EXTRA_PNGINFO".to_owned(),
                    value: extra_pnginfo,
                },
            );
        }
        inputs
    }

    fn interpolation_inputs_for_test(
        model: NativeOpaqueHandle,
        images: NativeOpaqueHandle,
        multiplier: NativePrimitive,
    ) -> BTreeMap<String, NativeValue> {
        BTreeMap::from([
            (
                "interp_model".to_owned(),
                NativeValue::Handle { value: model },
            ),
            ("images".to_owned(), NativeValue::Handle { value: images }),
            (
                "multiplier".to_owned(),
                NativeValue::Primitive { value: multiplier },
            ),
        ])
    }

    #[test]
    fn create_video_descriptor_and_fraction_match_source() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture["feature_id"], FEATURE_ID);
        assert_eq!(
            fixture["source"]["sha256"],
            "db1d0d40e065a50a4b10c2780511e4a9a75482916a76c175000e58ab875da7c9"
        );
        assert_eq!(
            exact_positive_f64_fraction(29.97),
            Some((1_054_475_631_502_295, 35_184_372_088_832))
        );
        assert_eq!(exact_positive_f64_fraction(30.0), Some((30, 1)));
        let binding = native_node_binding()?;
        binding.validate()?;
        let descriptor = binding.descriptor();
        assert_eq!(descriptor.class_type, CLASS_TYPE);
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["images", "fps", "audio", "bit_depth"]
        );
        assert_eq!(descriptor.outputs[0].name, "video");
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        let schema = descriptor
            .source_schema
            .as_ref()
            .ok_or("missing source schema")?;
        assert_eq!(
            schema.inputs[1].default,
            Some(crate::NativeSchemaValue::FiniteDecimal {
                value: "30.0".to_owned()
            })
        );
        assert_eq!(
            schema.inputs[3].default,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 8 })
        );
        Ok(())
    }

    #[test]
    fn create_video_publishes_exact_component_aliases() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let (image, image_storage) = harness.image_handle()?;
        let (audio, audio_storage) = harness.audio_handle()?;
        let outcome = futures::executor::block_on(executable()?.execute(
            harness.context(CancellationToken::default())?,
            harness.inputs(
                image,
                29.97,
                Some(audio),
                Some(NativePrimitive::UnsignedInteger(10)),
            ),
        ))?;
        let video = harness.video(outcome)?;
        let components = video.components().ok_or("component VIDEO was not retained")?;
        assert_eq!(components.frames().storage_id(), image_storage);
        assert_eq!(
            video.frame_rate(),
            (1_054_475_631_502_295, 35_184_372_088_832)
        );
        assert_eq!(video.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            components
                .audio()
                .ok_or("audio was not retained")?
                .waveform()
                .storage_id(),
            audio_storage
        );
        assert!(components.alpha().is_none());
        assert!(components.metadata().is_empty());
        assert_eq!(harness.store.count()?, 3);
        Ok(())
    }

    #[test]
    fn create_video_rejects_invalid_inputs_and_cancellation_without_publication()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let (image, _) = harness.image_handle()?;
        let node = executable()?;
        for (fps, depth) in [
            (0.0, Some(NativePrimitive::Integer(8))),
            (121.0, Some(NativePrimitive::Integer(8))),
            (f64::NAN, Some(NativePrimitive::Integer(8))),
            (30.0, Some(NativePrimitive::Integer(9))),
        ] {
            let error = futures::executor::block_on(node.execute(
                harness.context(CancellationToken::default())?,
                harness.inputs(image.clone(), fps, None, depth),
            ))
            .expect_err("invalid CreateVideo input must fail");
            assert_eq!(error.code, "invalid_node_inputs");
            assert_eq!(harness.store.count()?, 1);
        }

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(node.execute(
            harness.context(cancellation)?,
            harness.inputs(image.clone(), 30.0, None, None),
        ))
        .expect_err("cancelled CreateVideo must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, 1);

        let outcome = futures::executor::block_on(node.execute(
            harness.context(CancellationToken::default())?,
            harness.inputs(image, 1.0, None, None),
        ))?;
        let video = harness.video(outcome)?;
        assert_eq!(video.frame_rate(), (1, 1));
        assert_eq!(video.bit_depth(), NativeVideoBitDepth::Eight);
        assert!(video
            .components()
            .ok_or("component VIDEO was not retained")?
            .audio()
            .is_none());
        assert_eq!(harness.store.count()?, 2);
        Ok(())
    }

    #[test]
    fn get_video_components_descriptor_and_aliases_match_source() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(COMPONENTS_FIXTURE)?;
        assert_eq!(fixture["feature_id"], COMPONENTS_FEATURE_ID);
        assert_eq!(
            fixture["source"]["definition_sha256"],
            "b2232b2c558b472226418d56910b9b763d8398612b31c4019d4881d0c4717b8d"
        );
        let binding = components_node_binding()?;
        binding.validate()?;
        assert_eq!(binding.feature_id(), COMPONENTS_FEATURE_ID);
        let descriptor = binding.descriptor();
        assert_eq!(descriptor.class_type, COMPONENTS_CLASS_TYPE);
        assert_eq!(descriptor.inputs.len(), 1);
        assert_eq!(descriptor.inputs[0].name, "video");
        assert_eq!(
            descriptor
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["images", "audio", "fps", "bit_depth"]
        );
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        let NativeNodeBinding::Executable { presentation, .. } = binding else {
            return Err("GetVideoComponents binding is not executable".into());
        };
        assert_eq!(
            presentation.description,
            "Extracts all components from a video: frames, audio, framerate, and bit depth."
        );
        assert_eq!(
            presentation.search_aliases,
            ["extract frames", "split video", "video to images", "demux"]
        );
        Ok(())
    }

    #[test]
    fn get_video_components_preserves_aliases_and_nullable_audio() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let (image_handle, image_storage) = harness.image_handle()?;
        let image_payload = harness.store.payload(&image_handle)?;
        let NativeStoredPayload::Tensor(image_payload) = image_payload.as_ref() else {
            return Err("test IMAGE payload is unavailable".into());
        };
        let (audio_handle, audio_storage) = harness.audio_handle()?;
        let audio_payload = harness.store.payload(&audio_handle)?;
        let NativeStoredPayload::Audio(audio_payload) = audio_payload.as_ref() else {
            return Err("test AUDIO payload is unavailable".into());
        };
        let video = harness.video_handle(
            image_payload.tensor().clone(),
            Some(audio_payload.as_ref().clone()),
            (24_000, 1_001),
            NativeVideoBitDepth::Ten,
        )?;
        let outcome = futures::executor::block_on(components_executable()?.execute(
            harness.context(CancellationToken::default())?,
            BTreeMap::from([(
                "video".to_owned(),
                NativeValue::Handle {
                    value: video,
                },
            )]),
        ))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("GetVideoComponents did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let [
            NativeValue::Handle { value: image },
            NativeValue::Handle { value: audio },
            NativeValue::Primitive {
                value: NativePrimitive::Number(fps),
            },
            NativeValue::Primitive {
                value: NativePrimitive::Integer(depth),
            },
        ] = outputs.as_slice()
        else {
            return Err("GetVideoComponents returned the wrong output shape".into());
        };
        let image = harness.store.payload(image)?;
        let NativeStoredPayload::Tensor(image) = image.as_ref() else {
            return Err("GetVideoComponents IMAGE output is invalid".into());
        };
        let audio = harness.store.payload(audio)?;
        let NativeStoredPayload::Audio(audio) = audio.as_ref() else {
            return Err("GetVideoComponents AUDIO output is invalid".into());
        };
        assert_eq!(image.tensor().storage_id(), image_storage);
        assert_eq!(audio.waveform().storage_id(), audio_storage);
        assert_eq!(*fps, 24_000.0 / 1_001.0);
        assert_eq!(*depth, 10);

        let video = harness.video_handle(
            image_payload.tensor().clone(),
            None,
            (30, 1),
            NativeVideoBitDepth::Eight,
        )?;
        let outcome = futures::executor::block_on(components_executable()?.execute(
            harness.context(CancellationToken::default())?,
            BTreeMap::from([(
                "video".to_owned(),
                NativeValue::Handle {
                    value: video,
                },
            )]),
        ))?;
        let NativeNodeOutcome::Values { outputs, .. } = outcome else {
            return Err("GetVideoComponents did not return values".into());
        };
        assert!(matches!(
            outputs.get(1),
            Some(NativeValue::Primitive {
                value: NativePrimitive::Null
            })
        ));
        assert!(matches!(
            outputs.get(2),
            Some(NativeValue::Primitive {
                value: NativePrimitive::Number(value)
            }) if *value == 30.0
        ));
        assert!(matches!(
            outputs.get(3),
            Some(NativeValue::Primitive {
                value: NativePrimitive::Integer(8)
            })
        ));
        Ok(())
    }

    #[test]
    fn get_video_components_rejects_encoded_backing_without_decoding_or_publication()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let cancellation = CancellationToken::default();
        let execution = harness.backend.execution_context(
            StreamId::DEFAULT,
            harness.workspace.authorize_workspace(1024)?,
            &cancellation,
        );
        let image = ImageTensor::from_f32(
            &harness.backend,
            &execution,
            1,
            1,
            1,
            3,
            &[0.0, 0.5, 1.0],
        )?;
        let source = NativeVideoPayload::checked(
            image.tensor().clone(),
            30,
            1,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::new(),
        )?;
        let encoded_content = b"HMP4";
        let descriptor = TensorDescriptor::contiguous(
            vec![encoded_content.len() as u64],
            DType::U8,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (encoded_bytes, _) =
            harness
                .backend
                .upload_bytes(descriptor, encoded_content, &execution)?;
        let encoded = NativeVideoPayload::checked_h264_mp4_from_component(
            &source,
            encoded_bytes,
            Sha256::digest(encoded_content).into(),
            (1, 1),
            (30, 1),
            1,
        )?;
        let video = harness.store.publish(
            NativeStoredPayload::Video(Arc::new(encoded)),
            &CancellationToken::default(),
        )?;
        let count_before = harness.store.count()?;
        let failure = futures::executor::block_on(components_executable()?.execute(
            harness.context(CancellationToken::default())?,
            BTreeMap::from([("video".to_owned(), NativeValue::Handle { value: video })]),
        ))
        .expect_err("encoded VIDEO must not be projected as materialized components");
        assert_eq!(failure.code, "invalid_video_components");
        assert!(failure.message.contains("cannot decode an encoded VIDEO"));
        assert_eq!(harness.store.count()?, count_before);
        Ok(())
    }

    #[test]
    fn get_video_components_normalizes_u8_and_cancels_atomically() -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 1, 2, 3],
            DType::U8,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let cancellation = CancellationToken::default();
        let context = harness.backend.execution_context(
            StreamId::DEFAULT,
            harness.workspace.authorize_workspace(0)?,
            &cancellation,
        );
        let (frames, _) =
            harness
                .backend
                .upload_bytes(descriptor, &[0, 1, 127, 128, 254, 255], &context)?;
        let video = harness.video_handle(frames, None, (30, 1), NativeVideoBitDepth::Eight)?;
        let outcome = futures::executor::block_on(components_executable()?.execute(
            harness.context(CancellationToken::default())?,
            BTreeMap::from([(
                "video".to_owned(),
                NativeValue::Handle {
                    value: video.clone(),
                },
            )]),
        ))?;
        let NativeNodeOutcome::Values { outputs, .. } = outcome else {
            return Err("GetVideoComponents did not return values".into());
        };
        let Some(NativeValue::Handle { value: image }) = outputs.first() else {
            return Err("GetVideoComponents IMAGE output is absent".into());
        };
        let image = harness.store.payload(image)?;
        let NativeStoredPayload::Tensor(image) = image.as_ref() else {
            return Err("GetVideoComponents IMAGE output is invalid".into());
        };
        assert_eq!(
            image
                .image()
                .ok_or("IMAGE wrapper is absent")?
                .as_f32_slice()?,
            &[
                0.0,
                1.0 / 255.0,
                127.0 / 255.0,
                128.0 / 255.0,
                254.0 / 255.0,
                1.0
            ]
        );

        let baseline = harness.store.count()?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(components_executable()?.execute(
            harness.context(cancellation)?,
            BTreeMap::from([("video".to_owned(), NativeValue::Handle { value: video })]),
        ))
        .expect_err("cancelled GetVideoComponents must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, baseline);
        Ok(())
    }

    #[test]
    fn frame_interpolate_descriptor_fixture_and_cache_match_source() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(INTERPOLATION_FIXTURE)?;
        assert_eq!(fixture["feature_id"], INTERPOLATE_FEATURE_ID);
        assert_eq!(
            fixture["source"]["definition_sha256"],
            "e0b9dd6ec3b09e665bcc0f95d2b7a0209d9045ba9b96828e46f126e6914f049c"
        );
        let binding = interpolate_node_binding()?;
        binding.validate()?;
        assert_eq!(binding.feature_id(), INTERPOLATE_FEATURE_ID);
        let descriptor = binding.descriptor();
        assert_eq!(descriptor.class_type, INTERPOLATE_CLASS_TYPE);
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["interp_model", "images", "multiplier"]
        );
        assert_eq!(descriptor.outputs.len(), 1);
        assert_eq!(descriptor.outputs[0].name, "images");
        assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
        let schema = descriptor
            .source_schema
            .as_ref()
            .ok_or("FrameInterpolate source schema is absent")?;
        assert_eq!(
            schema.inputs[2].default,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 2 })
        );
        assert_eq!(
            schema.inputs[2].minimum,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 2 })
        );
        assert_eq!(
            schema.inputs[2].maximum,
            Some(crate::NativeSchemaValue::UnsignedInteger { value: 16 })
        );
        let NativeNodeBinding::Executable { presentation, .. } = binding else {
            return Err("FrameInterpolate binding is not executable".into());
        };
        assert_eq!(presentation.display_name, "Run Frame Interpolation Model");
        assert_eq!(presentation.category, "video");
        assert!(presentation.description.is_empty());
        assert_eq!(presentation.output_names, ["images"]);
        assert_eq!(
            presentation.search_aliases,
            [
                "rife",
                "film",
                "frame interpolation",
                "slow motion",
                "interpolate frames",
                "vfi"
            ]
        );
        Ok(())
    }

    #[test]
    fn frame_interpolate_bypass_returns_exact_input_handle_without_compute()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::interpolation()?;
        let (model, retained_model) = harness.interpolation_model_handle()?;
        let model_digest = retained_model.semantic_state_digest_sha256().to_owned();
        let (images, storage_id) = harness.image_handle_with_shape(1, 2, 2, 4, &[0.25; 16])?;
        let baseline = harness.store.count()?;
        let outcome = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context_without_compute(CancellationToken::default())?,
            interpolation_inputs_for_test(
                model,
                images.clone(),
                NativePrimitive::UnsignedInteger(2),
            ),
        ))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("FrameInterpolate bypass did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        assert_eq!(
            outputs,
            [NativeValue::Handle {
                value: images.clone()
            }]
        );
        assert_eq!(harness.store.count()?, baseline);
        let payload = harness.store.payload(&images)?;
        let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
            return Err("FrameInterpolate bypass IMAGE is invalid".into());
        };
        assert_eq!(payload.tensor().storage_id(), storage_id);
        assert_eq!(retained_model.semantic_state_digest_sha256(), model_digest);
        Ok(())
    }

    #[test]
    fn frame_interpolate_executes_reduced_rife_and_publishes_canonical_image()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::interpolation()?;
        let (model, retained_model) = harness.interpolation_model_handle()?;
        let model_digest = retained_model.semantic_state_digest_sha256().to_owned();
        let frame_elements = 64 * 64 * 3;
        let mut values = vec![0.0; frame_elements];
        values.extend(std::iter::repeat_n(1.0, frame_elements));
        let (images, input_storage) = harness.image_handle_with_shape(2, 64, 64, 3, &values)?;
        let baseline = harness.store.count()?;
        let outcome = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(CancellationToken::default())?,
            interpolation_inputs_for_test(model, images, NativePrimitive::UnsignedInteger(2)),
        ))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui,
            effects,
        } = outcome
        else {
            return Err("FrameInterpolate did not return values".into());
        };
        assert!(ui.is_none());
        assert!(effects.is_empty());
        let [NativeValue::Handle { value: output }] = outputs.as_slice() else {
            return Err("FrameInterpolate returned the wrong output shape".into());
        };
        let payload = harness.store.payload(output)?;
        let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
            return Err("FrameInterpolate output is not an IMAGE payload".into());
        };
        let image = payload
            .image()
            .ok_or("FrameInterpolate output has no canonical ImageTensor")?;
        assert_eq!(image.dimensions()?, (3, 64, 64, 3));
        assert_eq!(image.tensor().descriptor().dtype(), DType::F32);
        assert_eq!(image.tensor().descriptor().device(), DeviceId::CPU);
        assert_eq!(image.tensor().descriptor().stream(), StreamId::DEFAULT);
        assert!(image.tensor().descriptor().is_contiguous()?);
        assert_ne!(image.tensor().storage_id(), input_storage);
        let output_values = image.as_f32_slice()?;
        assert_eq!(output_values.first(), Some(&0.0));
        assert_eq!(output_values.get(frame_elements), Some(&0.5));
        assert_eq!(output_values.last(), Some(&1.0));
        assert_eq!(harness.store.count()?, baseline + 1);
        assert_eq!(retained_model.semantic_state_digest_sha256(), model_digest);
        Ok(())
    }

    #[test]
    fn frame_interpolate_rejects_invalid_requests_and_cancellation_atomically()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::interpolation()?;
        let (model, _) = harness.interpolation_model_handle()?;
        let (images, _) = harness.image_handle_with_shape(2, 2, 2, 4, &[0.5; 32])?;
        let baseline = harness.store.count()?;
        let error = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(CancellationToken::default())?,
            interpolation_inputs_for_test(
                model.clone(),
                images.clone(),
                NativePrimitive::UnsignedInteger(17),
            ),
        ))
        .expect_err("out-of-range multiplier must fail");
        assert_eq!(error.code, "invalid_node_inputs");
        let error = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(CancellationToken::default())?,
            interpolation_inputs_for_test(
                model.clone(),
                images.clone(),
                NativePrimitive::UnsignedInteger(2),
            ),
        ))
        .expect_err("non-bypass RGBA input must fail");
        assert_eq!(error.code, "invalid_node_inputs");
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let error = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(cancellation)?,
            interpolation_inputs_for_test(model, images, NativePrimitive::UnsignedInteger(2)),
        ))
        .expect_err("cancelled FrameInterpolate must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, baseline);
        Ok(())
    }

    #[test]
    fn frame_interpolate_maps_exhaustion_rolls_back_late_cancellation_and_retries()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::interpolation()?;
        let (model, _) = harness.interpolation_model_handle()?;
        let frame_elements = 64 * 64 * 3;
        let mut values = vec![0.0; frame_elements];
        values.extend(std::iter::repeat_n(1.0, frame_elements));
        let (images, _) = harness.image_handle_with_shape(2, 64, 64, 3, &values)?;
        let baseline_handles = harness.store.count()?;
        let baseline_memory = harness.workspace.memory_snapshot().current_bytes;

        let error = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context_with_scratch(CancellationToken::default(), 1_024)?,
            interpolation_inputs_for_test(
                model.clone(),
                images.clone(),
                NativePrimitive::UnsignedInteger(2),
            ),
        ))
        .expect_err("constrained FrameInterpolate must fail");
        assert_eq!(error.code, "frame_interpolation_resource_exhausted");
        assert!(!error.retryable);
        assert_eq!(harness.store.count()?, baseline_handles);
        assert_eq!(
            harness.workspace.memory_snapshot().current_bytes,
            baseline_memory
        );

        let cancellation = CancellationToken::default();
        harness.store.cancel_after_next_publish();
        let error = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(cancellation)?,
            interpolation_inputs_for_test(
                model.clone(),
                images.clone(),
                NativePrimitive::UnsignedInteger(2),
            ),
        ))
        .expect_err("late-cancelled FrameInterpolate must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert_eq!(harness.store.count()?, baseline_handles);
        assert_eq!(
            harness.workspace.memory_snapshot().current_bytes,
            baseline_memory
        );

        let outcome = futures::executor::block_on(interpolation_executable()?.execute(
            harness.context(CancellationToken::default())?,
            interpolation_inputs_for_test(model, images, NativePrimitive::UnsignedInteger(2)),
        ))?;
        assert!(matches!(outcome, NativeNodeOutcome::Values { .. }));
        assert_eq!(harness.store.count()?, baseline_handles + 1);
        assert!(harness.workspace.memory_snapshot().current_bytes > baseline_memory);
        Ok(())
    }

    #[test]
    fn save_webm_descriptor_and_fixture_match_source() -> Result<(), Box<dyn Error>> {
        let fixture: Value = serde_json::from_str(SAVE_WEBM_FIXTURE)?;
        assert_eq!(fixture["feature_id"], SAVE_WEBM_FEATURE_ID);
        assert_eq!(fixture["source"]["definition_sha256"], "55496b10af66a908ef035d236f8fab8193c1ae44408dab9d202deadff3be2715");
        let binding = save_webm_node_binding()?;
        binding.validate()?;
        let descriptor = binding.descriptor();
        assert_eq!(descriptor.class_type, SAVE_WEBM_CLASS_TYPE);
        assert!(descriptor.output_node);
        assert_eq!(descriptor.effect, NativeEffectClass::WritesArtifact);
        assert_eq!(descriptor.cache, NativeCachePolicy::Never);
        assert_eq!(descriptor.outputs[0].name, "images");
        assert_eq!(
            descriptor
                .inputs
                .iter()
                .map(|input| (input.name.as_str(), input.hidden))
                .collect::<Vec<_>>(),
            [
                ("images", false),
                ("filename_prefix", false),
                ("codec", false),
                ("fps", false),
                ("crf", false),
                ("prompt", true),
                ("extra_pnginfo", true),
            ]
        );
        let schema = descriptor
            .source_schema
            .as_ref()
            .ok_or("SaveWEBM source schema is absent")?;
        assert_eq!(schema.inputs[2].choices.len(), 2);
        assert_eq!(schema.inputs[3].minimum, Some(crate::NativeSchemaValue::FiniteDecimal { value: "0.01".to_owned() }));
        let NativeNodeBinding::Executable { presentation, .. } = binding else {
            return Err("SaveWEBM binding is not executable".into());
        };
        assert_eq!(presentation.display_name, "Save WEBM");
        assert_eq!(presentation.search_aliases, ["export webm"]);
        assert!(presentation.is_experimental);
        Ok(())
    }

    #[test]
    fn save_webm_preserves_metadata_image_identity_and_prepares_video_output()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let (images, storage_id) = harness.image_handle_with_shape(2, 2, 2, 4, &[0.5; 32])?;
        let baseline_handles = harness.store.count()?;
        let (context, webm, effects) =
            harness.save_webm_context(CancellationToken::default(), false)?;
        let inputs = save_webm_test_inputs(
            images.clone(),
            "vp9",
            29.97,
            31.5,
            Some(json!({"1": {"class_type": "SaveWEBM"}})),
            Some(json!({"workflow": {"version": 1}, "prompt": "replacement"})),
        );
        let outcome = futures::executor::block_on(save_webm_executable()?.execute(context, inputs))?;
        let NativeNodeOutcome::Values {
            outputs,
            ui: Some(ui),
            effects: prepared,
        } = outcome
        else {
            return Err("SaveWEBM did not return values, UI, and a prepared effect".into());
        };
        assert_eq!(outputs, [NativeValue::Handle { value: images.clone() }]);
        assert_eq!(prepared.len(), 1);
        assert_eq!(ui["animated"], json!([true]));
        assert_eq!(ui["images"][0]["batch_index"], 0);
        assert_eq!(ui["images"][0]["type"], "output");
        assert_eq!(ui["images"][0]["transaction_id"], prepared[0].transaction_id().to_string());
        assert_eq!(harness.store.count()?, baseline_handles);
        let payload = harness.store.payload(&images)?;
        let NativeStoredPayload::Tensor(payload) = payload.as_ref() else {
            return Err("SaveWEBM input payload changed type".into());
        };
        assert_eq!(payload.tensor().storage_id(), storage_id);

        let requests = webm
            .requests
            .lock()
            .map_err(|_| "WebM request lock was poisoned")?;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, NativeVideoCodec::Vp9);
        assert_eq!(requests[0].1, (2_997, 100));
        assert_eq!(requests[0].2, 31.5_f64.to_bits());
        assert_eq!(
            requests[0].3,
            [
                ("prompt".to_owned(), "{\"1\":{\"class_type\":\"SaveWEBM\"}}".to_owned()),
                ("workflow".to_owned(), "{\"version\":1}".to_owned()),
                ("prompt".to_owned(), "\"replacement\"".to_owned()),
            ]
        );
        drop(requests);
        let output_requests = effects
            .requests
            .lock()
            .map_err(|_| "effect request lock was poisoned")?;
        assert_eq!(output_requests.len(), 1);
        assert_eq!(output_requests[0].namespace(), NativeOutputNamespace::Output);
        assert_eq!(output_requests[0].filename_prefix(), "video/ComfyUI");
        assert_eq!(output_requests[0].extension(), "webm");
        assert_eq!(output_requests[0].media_type(), "video/webm");
        assert_eq!(output_requests[0].media_kind(), NativeOutputMediaKind::Video);
        assert_eq!(output_requests[0].shape(), NativeOutputShape::File);
        assert_eq!(output_requests[0].content().as_ref(), b"HPPPT");
        Ok(())
    }

    #[test]
    fn save_webm_rejects_invalid_inputs_rolls_back_late_cancellation_and_retries()
    -> Result<(), Box<dyn Error>> {
        let harness = Harness::new()?;
        let (images, _) = harness.image_handle()?;
        let invalid = save_webm_test_inputs(
            images.clone(),
            "h264",
            24.0,
            32.0,
            None,
            None,
        );
        let (context, _, _) =
            harness.save_webm_context(CancellationToken::default(), false)?;
        let error = futures::executor::block_on(save_webm_executable()?.execute(context, invalid))
            .expect_err("invalid SaveWEBM codec must fail");
        assert_eq!(error.code, "invalid_save_webm");

        let cancellation = CancellationToken::default();
        let (context, _, effects) = harness.save_webm_context(cancellation, true)?;
        let inputs = save_webm_test_inputs(
            images.clone(),
            "av1",
            24.0,
            32.0,
            None,
            Some(json!({"workflow": {}})),
        );
        let error = futures::executor::block_on(save_webm_executable()?.execute(context, inputs))
            .expect_err("late-cancelled SaveWEBM must fail");
        assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        assert!(effects
            .prepared
            .lock()
            .map_err(|_| "prepared effect lock was poisoned")?
            .is_empty());

        let (context, _, _) =
            harness.save_webm_context(CancellationToken::default(), false)?;
        let retry = save_webm_test_inputs(images, "av1", 24.0, 32.0, None, None);
        let outcome = futures::executor::block_on(save_webm_executable()?.execute(context, retry))?;
        assert!(matches!(outcome, NativeNodeOutcome::Values { .. }));
        Ok(())
    }
}
