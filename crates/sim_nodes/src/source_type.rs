use crate::{
    NativeHandleKind, NativeHandleType, NativeInputSchemaMetadata, NativeNodeContractError,
    NativeOutputSchemaMetadata, NativePrimitiveType, NativeSchemaField, NativeSchemaValue,
    NativeTypeUnion, NativeValueType,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSourceTypeOwner {
    Inline,
    Compute,
    MediaAsset,
    Provider,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSourceValueClass {
    Any,
    Primitive(NativePrimitiveType),
    PreservedUnknown,
    SchemaScalar,
    Handle(NativeHandleKind),
}

pub fn native_handle_type_accepts(expected: &NativeHandleType, actual: &NativeHandleType) -> bool {
    if expected == actual {
        return true;
    }
    if expected.kind != NativeHandleKind::ThreeD || actual.kind != NativeHandleKind::ThreeD {
        return false;
    }
    match expected.type_id.as_str() {
        "FILE_3D" => matches!(
            actual.type_id.as_str(),
            "FILE_3D_FBX"
                | "FILE_3D_GLTF"
                | "FILE_3D_GLB"
                | "FILE_3D_KSPLAT"
                | "FILE_3D_OBJ"
                | "FILE_3D_PLY"
                | "FILE_3D_SPLAT"
                | "FILE_3D_SPZ"
                | "FILE_3D_STL"
                | "FILE_3D_USDZ"
        ),
        "FILE_3D_POINT_CLOUD_ANY" => actual.type_id == "FILE_3D_PLY",
        "FILE_3D_SPLAT_ANY" => matches!(
            actual.type_id.as_str(),
            "FILE_3D_PLY" | "FILE_3D_SPZ" | "FILE_3D_SPLAT" | "FILE_3D_KSPLAT"
        ),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeSourceTypeProjection {
    source_type: &'static str,
    owner: NativeSourceTypeOwner,
    value_class: NativeSourceValueClass,
    role: &'static str,
    handle_type_id: Option<&'static str>,
}

impl NativeSourceTypeProjection {
    const fn inline(
        source_type: &'static str,
        value_class: NativeSourceValueClass,
        role: &'static str,
    ) -> Self {
        Self {
            source_type,
            owner: NativeSourceTypeOwner::Inline,
            value_class,
            role,
            handle_type_id: None,
        }
    }

    const fn handle(
        source_type: &'static str,
        owner: NativeSourceTypeOwner,
        kind: NativeHandleKind,
        role: &'static str,
        handle_type_id: &'static str,
    ) -> Self {
        Self {
            source_type,
            owner,
            value_class: NativeSourceValueClass::Handle(kind),
            role,
            handle_type_id: Some(handle_type_id),
        }
    }

    pub const fn source_type(&self) -> &'static str {
        self.source_type
    }

    pub const fn owner(&self) -> NativeSourceTypeOwner {
        self.owner
    }

    pub const fn value_class(&self) -> NativeSourceValueClass {
        self.value_class
    }

    pub const fn role(&self) -> &'static str {
        self.role
    }

    pub fn value_type(&self) -> Result<NativeValueType, NativeSourceTypeError> {
        match self.value_class {
            NativeSourceValueClass::Any => Ok(NativeValueType::Any),
            NativeSourceValueClass::Primitive(value) => Ok(NativeValueType::Primitive(value)),
            NativeSourceValueClass::PreservedUnknown => Ok(NativeValueType::NamedPreservedUnknown(
                self.source_type.to_owned(),
            )),
            NativeSourceValueClass::SchemaScalar => Err(NativeSourceTypeError::SchemaRequired(
                self.source_type.to_owned(),
            )),
            NativeSourceValueClass::Handle(kind) => {
                Ok(NativeValueType::Handle(NativeHandleType::new(
                    kind,
                    self.handle_type_id.ok_or_else(|| {
                        NativeSourceTypeError::InvalidProjection(self.source_type.to_owned())
                    })?,
                )?))
            }
        }
    }

    pub fn handle_type(&self) -> Result<Option<NativeHandleType>, NativeSourceTypeError> {
        match self.value_class {
            NativeSourceValueClass::Handle(kind) => Ok(Some(NativeHandleType::new(
                kind,
                self.handle_type_id.ok_or_else(|| {
                    NativeSourceTypeError::InvalidProjection(self.source_type.to_owned())
                })?,
            )?)),
            _ => Ok(None),
        }
    }
}

#[derive(Debug, Error)]
pub enum NativeSourceTypeError {
    #[error("native source type `{0}` is not registered")]
    Unknown(String),
    #[error("native source type `{0}` requires its exact preserved source identity")]
    SourceIdentityRequired(String),
    #[error("native source type `{0}` requires its port schema to resolve the scalar type")]
    SchemaRequired(String),
    #[error("native source type projection for `{0}` is invalid")]
    InvalidProjection(String),
    #[error(transparent)]
    Contract(#[from] NativeNodeContractError),
}

macro_rules! inline {
    ($source:literal, $class:expr, $role:literal) => {
        NativeSourceTypeProjection::inline($source, $class, $role)
    };
}

macro_rules! handle {
    ($source:literal, $owner:ident, $kind:ident, $role:literal) => {
        NativeSourceTypeProjection::handle(
            $source,
            NativeSourceTypeOwner::$owner,
            NativeHandleKind::$kind,
            $role,
            $source,
        )
    };
    ($source:literal, $owner:ident, $kind:ident, $role:literal, $type_id:literal) => {
        NativeSourceTypeProjection::handle(
            $source,
            NativeSourceTypeOwner::$owner,
            NativeHandleKind::$kind,
            $role,
            $type_id,
        )
    };
}

pub fn native_source_type_projection(
    source_type: &str,
) -> Result<NativeSourceTypeProjection, NativeSourceTypeError> {
    use NativePrimitiveType::{Boolean, Integer, Number, String as StringPrimitive};
    use NativeSourceValueClass::{Any, PreservedUnknown, Primitive, SchemaScalar};
    let projection = match source_type {
        "ANY" => inline!("ANY", Any, "any"),
        "*" => inline!("*", Any, "any_type"),
        "ARRAY" => inline!("ARRAY", PreservedUnknown, "array"),
        "BOOLEAN" => inline!("BOOLEAN", Primitive(Boolean), "boolean"),
        "BOUNDING_BOXES" => inline!("BOUNDING_BOXES", PreservedUnknown, "bounding_boxes"),
        "COLOR" => inline!("COLOR", Primitive(StringPrimitive), "color"),
        "COLORS" => inline!("COLORS", PreservedUnknown, "colors"),
        "COMBO" => inline!("COMBO", SchemaScalar, "choice"),
        "CROP_METHODS" => inline!("CROP_METHODS", Primitive(StringPrimitive), "crop_method"),
        "DICT" => inline!("DICT", PreservedUnknown, "dictionary"),
        "COMFY_DYNAMICCOMBO_V3" => inline!("COMFY_DYNAMICCOMBO_V3", SchemaScalar, "dynamic_choice"),
        "EXTRA_PNGINFO" => inline!("EXTRA_PNGINFO", PreservedUnknown, "extra_png_info"),
        "FLOAT" => inline!("FLOAT", Primitive(Number), "float"),
        "FLOATS" => inline!("FLOATS", PreservedUnknown, "float_list"),
        "GET_FILENAME_LIST" => inline!("GET_FILENAME_LIST", Primitive(StringPrimitive), "filename"),
        "IMAGECOMPARE" => inline!("IMAGECOMPARE", PreservedUnknown, "image_compare_state"),
        "INT" => inline!("INT", Primitive(Integer), "integer"),
        "COMFY_MATCHTYPE_V3" => inline!("COMFY_MATCHTYPE_V3", Any, "match_type"),
        "PATHS" => inline!("PATHS", Primitive(StringPrimitive), "path_selection"),
        "PROMPT" => inline!("PROMPT", PreservedUnknown, "prompt"),
        "SAMPLERS" => inline!("SAMPLERS", Primitive(StringPrimitive), "sampler_name"),
        "SCHEDULERS" => inline!("SCHEDULERS", Primitive(StringPrimitive), "scheduler_name"),
        "SORTED" => inline!("SORTED", Primitive(StringPrimitive), "sorted_filename"),
        "STRING" => inline!("STRING", Primitive(StringPrimitive), "string"),
        "TIMESTEPS_RANGE" => inline!("TIMESTEPS_RANGE", PreservedUnknown, "timesteps_range"),
        "UPSCALE_METHODS" => inline!(
            "UPSCALE_METHODS",
            Primitive(StringPrimitive),
            "upscale_method"
        ),
        "VAE_LIST" => inline!("VAE_LIST", Primitive(StringPrimitive), "vae_name"),
        "_COLOR_CHANNELS" => inline!(
            "_COLOR_CHANNELS",
            Primitive(StringPrimitive),
            "color_channel"
        ),
        "_LIST" => inline!("_LIST", Primitive(StringPrimitive), "source_list_choice"),

        "AUDIO_ENCODER" => handle!("AUDIO_ENCODER", Compute, Model, "audio_encoder"),
        "AUDIO_ENCODER_OUTPUT" => handle!(
            "AUDIO_ENCODER_OUTPUT",
            Compute,
            StructuredCompute,
            "audio_encoder_output"
        ),
        "BACKGROUND_REMOVAL" => handle!("BACKGROUND_REMOVAL", Compute, Model, "background_removal"),
        "BOUNDING_BOX" => handle!("BOUNDING_BOX", Compute, StructuredCompute, "bounding_box"),
        "CLIP" => handle!("CLIP", Compute, Clip, "clip"),
        "CLIP_VISION" => handle!("CLIP_VISION", Compute, Clip, "clip_vision"),
        "CLIP_VISION_OUTPUT" => handle!(
            "CLIP_VISION_OUTPUT",
            Compute,
            StructuredCompute,
            "clip_vision_output"
        ),
        "CONDITIONING" => handle!("CONDITIONING", Compute, Conditioning, "conditioning"),
        "CONTROL_NET" => handle!("CONTROL_NET", Compute, ControlNet, "controlnet"),
        "CURVE" => inline!("CURVE", PreservedUnknown, "curve"),
        "DA3_MODEL" => handle!("DA3_MODEL", Compute, Model, "da3_model"),
        "FACE_DETECTION_MODEL" => {
            handle!("FACE_DETECTION_MODEL", Compute, Model, "face_detection")
        }
        "FACE_LANDMARKS" => handle!(
            "FACE_LANDMARKS",
            Compute,
            StructuredCompute,
            "face_landmarks"
        ),
        "INTERP_MODEL" => handle!("INTERP_MODEL", Compute, Model, "frame_interpolation"),
        "GLIGEN" => handle!("GLIGEN", Compute, Model, "gligen"),
        "GUIDER" => handle!("GUIDER", Compute, StructuredCompute, "guider"),
        "HISTOGRAM" => inline!("HISTOGRAM", PreservedUnknown, "histogram"),
        "HOOKS" => handle!("HOOKS", Compute, Model, "hooks"),
        "HOOK_KEYFRAMES" => handle!("HOOK_KEYFRAMES", Compute, Model, "hook_keyframes"),
        "IC_LORA_PARAMETERS" => handle!(
            "IC_LORA_PARAMETERS",
            Compute,
            StructuredCompute,
            "ic_lora_parameters"
        ),
        "LATENT" => handle!("LATENT", Compute, Latent, "latent"),
        "LATENT_OPERATION" => handle!("LATENT_OPERATION", Compute, Model, "latent_operation"),
        "LATENT_UPSCALE_MODEL" => {
            handle!(
                "LATENT_UPSCALE_MODEL",
                Compute,
                Model,
                "latent_upscale_model"
            )
        }
        "LORA_MODEL" => handle!("LORA_MODEL", Compute, Model, "lora_model"),
        "LOSS_MAP" => handle!("LOSS_MAP", Compute, StructuredCompute, "loss_map"),
        "MODEL" => handle!("MODEL", Compute, Model, "model"),
        "MODEL_PATCH" => handle!("MODEL_PATCH", Compute, Model, "model_patch"),
        "MOGE_MODEL" => handle!("MOGE_MODEL", Compute, Model, "moge_model"),
        "NOISE" => handle!("NOISE", Compute, StructuredCompute, "noise"),
        "OPTICAL_FLOW" => handle!("OPTICAL_FLOW", Compute, Model, "optical_flow"),
        "PHOTOMAKER" => handle!("PHOTOMAKER", Compute, Model, "photomaker"),
        "POSE_KEYPOINT" => handle!("POSE_KEYPOINT", Compute, StructuredCompute, "pose_keypoint"),
        "SAM3_TRACK_DATA" => handle!(
            "SAM3_TRACK_DATA",
            Compute,
            StructuredCompute,
            "sam3_track_data"
        ),
        "SAMPLER" => handle!("SAMPLER", Compute, StructuredCompute, "sampler"),
        "SIGMAS" => handle!("SIGMAS", Compute, Tensor, "sigmas"),
        "STYLE_MODEL" => handle!("STYLE_MODEL", Compute, Model, "style_model"),
        "TRACKS" => handle!("TRACKS", Compute, StructuredCompute, "tracks"),
        "UPSCALE_MODEL" => handle!("UPSCALE_MODEL", Compute, Model, "upscale_model"),
        "VAE" => handle!("VAE", Compute, Vae, "vae"),
        "WAN_CAMERA_EMBEDDING" => handle!(
            "WAN_CAMERA_EMBEDDING",
            Compute,
            Tensor,
            "wan_camera_embedding"
        ),

        "AUDIO" => handle!("AUDIO", MediaAsset, Audio, "audio"),
        "AUDIO_RECORD" => handle!("AUDIO_RECORD", MediaAsset, Artifact, "audio_record"),
        "CAMERA_CONTROL" => handle!("CAMERA_CONTROL", MediaAsset, ThreeD, "camera_control"),
        "DA3_GEOMETRY" => handle!("DA3_GEOMETRY", MediaAsset, ThreeD, "da3_geometry"),
        "FILE_3D" => handle!("FILE_3D", MediaAsset, ThreeD, "file_any"),
        "FILE_3D_FBX" => handle!("FILE_3D_FBX", MediaAsset, ThreeD, "file_fbx"),
        "FILE_3D_GLTF" => handle!("FILE_3D_GLTF", MediaAsset, ThreeD, "file_gltf"),
        "FILE_3D_GLB" => handle!("FILE_3D_GLB", MediaAsset, ThreeD, "file_glb"),
        "FILE_3D_KSPLAT" => handle!("FILE_3D_KSPLAT", MediaAsset, ThreeD, "file_ksplat"),
        "FILE_3D_OBJ" => handle!("FILE_3D_OBJ", MediaAsset, ThreeD, "file_obj"),
        "FILE_3D_PLY" => handle!("FILE_3D_PLY", MediaAsset, ThreeD, "file_ply"),
        "FILE_3D_POINT_CLOUD_ANY" => handle!(
            "FILE_3D_POINT_CLOUD_ANY",
            MediaAsset,
            ThreeD,
            "file_point_cloud"
        ),
        "FILE_3D_SPLAT_ANY" => handle!("FILE_3D_SPLAT_ANY", MediaAsset, ThreeD, "file_splat"),
        "FILE_3D_SPLAT" => handle!("FILE_3D_SPLAT", MediaAsset, ThreeD, "file_splat_payload"),
        "FILE_3D_SPZ" => handle!("FILE_3D_SPZ", MediaAsset, ThreeD, "file_spz"),
        "FILE_3D_STL" => handle!("FILE_3D_STL", MediaAsset, ThreeD, "file_stl"),
        "FILE_3D_USDZ" => handle!("FILE_3D_USDZ", MediaAsset, ThreeD, "file_usdz"),
        "IMAGE" => handle!("IMAGE", MediaAsset, Image, "image"),
        "LOAD_3D" => handle!("LOAD_3D", MediaAsset, ThreeD, "load_3d"),
        "LOAD3D_CAMERA" => handle!("LOAD3D_CAMERA", MediaAsset, ThreeD, "camera"),
        "LOAD3D_MODEL_INFO" => handle!("LOAD3D_MODEL_INFO", MediaAsset, ThreeD, "model_info"),
        "MASK" => handle!("MASK", MediaAsset, Mask, "mask"),
        "MESH" => handle!("MESH", MediaAsset, ThreeD, "mesh"),
        "MOGE_GEOMETRY" => handle!("MOGE_GEOMETRY", MediaAsset, ThreeD, "moge_geometry"),
        "SPLAT" => handle!("SPLAT", MediaAsset, ThreeD, "splat"),
        "SVG" => handle!("SVG", MediaAsset, Artifact, "svg"),
        "VIDEO" => handle!("VIDEO", MediaAsset, Video, "video"),
        "VOXEL" => handle!("VOXEL", MediaAsset, ThreeD, "voxel"),
        "WEBCAM" => handle!("WEBCAM", MediaAsset, Artifact, "webcam"),

        "CUSTOM" => {
            return Err(NativeSourceTypeError::SourceIdentityRequired(
                source_type.to_owned(),
            ));
        }
        "GEMINI_INPUT_FILES" => handle!(
            "GEMINI_INPUT_FILES",
            Provider,
            ProviderTask,
            "gemini_input_files"
        ),
        "MESHY_RIGGED_TASK_ID" => handle!(
            "MESHY_RIGGED_TASK_ID",
            Provider,
            ProviderTask,
            "meshy_rigged_task"
        ),
        "MESHY_TASK_ID" => handle!("MESHY_TASK_ID", Provider, ProviderTask, "meshy_task"),
        "MODEL_TASK_ID" => handle!("MODEL_TASK_ID", Provider, ProviderTask, "model_task"),
        "MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID" => handle!(
            "MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID",
            Provider,
            ProviderTask,
            "tripo_model_task_union",
            "PROVIDER_TASK_TRIPO_MODEL_RIG_RETARGET"
        ),
        "OPENAI_CHAT_CONFIG" => handle!(
            "OPENAI_CHAT_CONFIG",
            Provider,
            ProviderTask,
            "openai_chat_config"
        ),
        "OPENAI_INPUT_FILES" => handle!(
            "OPENAI_INPUT_FILES",
            Provider,
            ProviderTask,
            "openai_input_files"
        ),
        "RETARGET_TASK_ID" => handle!("RETARGET_TASK_ID", Provider, ProviderTask, "retarget_task"),
        "RIG_TASK_ID" => handle!("RIG_TASK_ID", Provider, ProviderTask, "rig_task"),
        value => return Err(NativeSourceTypeError::Unknown(value.to_owned())),
    };
    Ok(projection)
}

pub fn native_custom_source_type_projection(
    source_identity: &str,
) -> Result<NativeSourceTypeProjection, NativeSourceTypeError> {
    let (role, handle_type_id) = match source_identity {
        "ELEVENLABS_VOICE" => ("elevenlabs_voice", "CUSTOM_ELEVENLABS_VOICE"),
        "KreaIO.STYLE_REF" => ("krea_style_ref", "CUSTOM_KREA_STYLE_REF"),
        "LumaIO.LUMA_CONCEPTS" => ("luma_concepts", "CUSTOM_LUMA_CONCEPTS"),
        "LumaIO.LUMA_REF" => ("luma_ref", "CUSTOM_LUMA_REF"),
        "LumaIO.LUMA_RAY32_KEYFRAME" => ("luma_ray32_keyframe", "CUSTOM_LUMA_RAY32_KEYFRAME"),
        "PixverseIO.TEMPLATE" => ("pixverse_template", "CUSTOM_PIXVERSE_TEMPLATE"),
        "RecraftIO.COLOR" => ("recraft_color", "CUSTOM_RECRAFT_COLOR"),
        "RecraftIO.STYLEV3" => ("recraft_style_v3", "CUSTOM_RECRAFT_STYLE_V3"),
        "RecraftIO.CONTROLS" => ("recraft_controls", "CUSTOM_RECRAFT_CONTROLS"),
        "RunwayAleph2IO.KEYFRAME" => ("runway_aleph2_keyframe", "CUSTOM_RUNWAY_ALEPH2_KEYFRAME"),
        "RunwayAleph2IO.PROMPT_IMAGE" => (
            "runway_aleph2_prompt_image",
            "CUSTOM_RUNWAY_ALEPH2_PROMPT_IMAGE",
        ),
        value => return Err(NativeSourceTypeError::Unknown(value.to_owned())),
    };
    Ok(NativeSourceTypeProjection::handle(
        "CUSTOM",
        NativeSourceTypeOwner::Provider,
        NativeHandleKind::ProviderTask,
        role,
        handle_type_id,
    ))
}

pub fn native_plugin_source_type_projection(
    plugin_type_name: &str,
) -> Result<NativeSourceTypeProjection, NativeSourceTypeError> {
    let custom_identity = match plugin_type_name {
        "custom-elevenlabs-voice" => Some("ELEVENLABS_VOICE"),
        "custom-krea-style-ref" => Some("KreaIO.STYLE_REF"),
        "custom-luma-concepts" => Some("LumaIO.LUMA_CONCEPTS"),
        "custom-luma-ray32-keyframe" => Some("LumaIO.LUMA_RAY32_KEYFRAME"),
        "custom-luma-ref" => Some("LumaIO.LUMA_REF"),
        "custom-pixverse-template" => Some("PixverseIO.TEMPLATE"),
        "custom-recraft-color" => Some("RecraftIO.COLOR"),
        "custom-recraft-controls" => Some("RecraftIO.CONTROLS"),
        "custom-recraft-style-v3" => Some("RecraftIO.STYLEV3"),
        "custom-runway-aleph2-keyframe" => Some("RunwayAleph2IO.KEYFRAME"),
        "custom-runway-aleph2-prompt-image" => Some("RunwayAleph2IO.PROMPT_IMAGE"),
        _ => None,
    };
    if let Some(custom_identity) = custom_identity {
        return native_custom_source_type_projection(custom_identity);
    }
    let source_type = match plugin_type_name {
        "any" => "*".to_owned(),
        "bounding-box-editor" => "BOUNDING_BOXES".to_owned(),
        "color-list" => "COLORS".to_owned(),
        "dictionary" => "DICT".to_owned(),
        "dynamic-combo" => "COMFY_DYNAMICCOMBO_V3".to_owned(),
        "file-3d-any" => "FILE_3D".to_owned(),
        "file-3d-point-cloud" => "FILE_3D_POINT_CLOUD_ANY".to_owned(),
        "file-3d-splat" => "FILE_3D_SPLAT_ANY".to_owned(),
        "float-list" => "FLOATS".to_owned(),
        "image-compare" => "IMAGECOMPARE".to_owned(),
        "integer" => "INT".to_owned(),
        "load-3d-camera" => "LOAD3D_CAMERA".to_owned(),
        "load-3d-model-info" => "LOAD3D_MODEL_INFO".to_owned(),
        "match-type" => "COMFY_MATCHTYPE_V3".to_owned(),
        "camera-control" => "CAMERA_CONTROL".to_owned(),
        "gemini-input-files" => "GEMINI_INPUT_FILES".to_owned(),
        "meshy-rigged-task-id" => "MESHY_RIGGED_TASK_ID".to_owned(),
        "meshy-task-id" => "MESHY_TASK_ID".to_owned(),
        "model-task-id" => "MODEL_TASK_ID".to_owned(),
        "openai-chat-config" => "OPENAI_CHAT_CONFIG".to_owned(),
        "openai-input-files" => "OPENAI_INPUT_FILES".to_owned(),
        "retarget-task-id" => "RETARGET_TASK_ID".to_owned(),
        "rig-task-id" => "RIG_TASK_ID".to_owned(),
        "autogrow" | "custom" | "multi-type" => {
            return Err(NativeSourceTypeError::SourceIdentityRequired(
                plugin_type_name.to_owned(),
            ));
        }
        value => value.replace('-', "_").to_ascii_uppercase(),
    };
    native_source_type_projection(&source_type)
}

pub fn native_value_types_for_input_schema(
    schema: &NativeInputSchemaMetadata,
) -> Result<NativeTypeUnion, NativeSourceTypeError> {
    let mut value_types = BTreeSet::new();
    for source_type in &schema.source_type_names {
        if source_type == "CUSTOM" {
            value_types.insert(
                native_custom_source_type_projection(source_identity(&schema.extra)?)?
                    .value_type()?,
            );
        } else if source_type == "COMBO" {
            value_types.extend(schema_scalar_types(
                &schema.choices,
                schema.default.as_ref(),
                "COMBO",
            ));
        } else if source_type == "COMFY_DYNAMICCOMBO_V3" {
            value_types.insert(NativeValueType::NamedPreservedUnknown(
                source_type.to_owned(),
            ));
        } else {
            value_types.insert(native_source_type_projection(source_type)?.value_type()?);
        }
    }
    Ok(NativeTypeUnion::new(value_types)?)
}

pub fn native_value_type_for_output_schema(
    schema: &NativeOutputSchemaMetadata,
) -> Result<NativeValueType, NativeSourceTypeError> {
    match schema.source_type_name.as_str() {
        "CUSTOM" => {
            native_custom_source_type_projection(source_identity(&schema.extra)?)?.value_type()
        }
        "COMBO" => {
            let values = schema_scalar_types(&schema.choices, None, "COMBO");
            if values.len() == 1 {
                values
                    .into_iter()
                    .next()
                    .ok_or_else(|| NativeSourceTypeError::SchemaRequired("COMBO".to_owned()))
            } else {
                Ok(NativeValueType::NamedPreservedUnknown("COMBO".to_owned()))
            }
        }
        "COMFY_DYNAMICCOMBO_V3" => Ok(NativeValueType::NamedPreservedUnknown(
            "COMFY_DYNAMICCOMBO_V3".to_owned(),
        )),
        source_type => native_source_type_projection(source_type)?.value_type(),
    }
}

fn source_identity(fields: &[NativeSchemaField]) -> Result<&str, NativeSourceTypeError> {
    let identities = fields
        .iter()
        .filter_map(|field| {
            if field.name == "source_identity"
                && let NativeSchemaValue::String { value } = &field.value
            {
                Some(value.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let [identity] = identities.as_slice() {
        Ok(identity)
    } else {
        Err(NativeSourceTypeError::SourceIdentityRequired(
            "CUSTOM".to_owned(),
        ))
    }
}

fn schema_scalar_types(
    choices: &[NativeSchemaValue],
    default: Option<&NativeSchemaValue>,
    preserved_type: &str,
) -> BTreeSet<NativeValueType> {
    let mut result = BTreeSet::new();
    for choice in choices.iter().chain(default) {
        result.insert(match choice {
            NativeSchemaValue::Null => NativeValueType::Primitive(NativePrimitiveType::Null),
            NativeSchemaValue::Boolean { .. } => {
                NativeValueType::Primitive(NativePrimitiveType::Boolean)
            }
            NativeSchemaValue::SignedInteger { .. } | NativeSchemaValue::UnsignedInteger { .. } => {
                NativeValueType::Primitive(NativePrimitiveType::Integer)
            }
            NativeSchemaValue::FiniteDecimal { .. } => {
                NativeValueType::Primitive(NativePrimitiveType::Number)
            }
            NativeSchemaValue::String { .. } => {
                NativeValueType::Primitive(NativePrimitiveType::String)
            }
            NativeSchemaValue::List { .. }
            | NativeSchemaValue::Object { .. }
            | NativeSchemaValue::PreservedExpression { .. } => {
                NativeValueType::NamedPreservedUnknown(preserved_type.to_owned())
            }
        });
    }
    if result.is_empty() {
        result.insert(NativeValueType::NamedPreservedUnknown(
            preserved_type.to_owned(),
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeOutputSchemaMetadata, NodeRegistry};
    use std::{collections::BTreeSet, error::Error};

    const PORTABLE_SOURCE_TYPES: &[&str] = &[
        "*",
        "ANY",
        "ARRAY",
        "AUDIO",
        "AUDIO_ENCODER",
        "AUDIO_ENCODER_OUTPUT",
        "AUDIO_RECORD",
        "BACKGROUND_REMOVAL",
        "BOOLEAN",
        "BOUNDING_BOX",
        "BOUNDING_BOXES",
        "CAMERA_CONTROL",
        "CLIP",
        "CLIP_VISION",
        "CLIP_VISION_OUTPUT",
        "COLOR",
        "COLORS",
        "COMBO",
        "COMFY_DYNAMICCOMBO_V3",
        "COMFY_MATCHTYPE_V3",
        "CONDITIONING",
        "CONTROL_NET",
        "CROP_METHODS",
        "CURVE",
        "CUSTOM",
        "DA3_GEOMETRY",
        "DA3_MODEL",
        "DICT",
        "EXTRA_PNGINFO",
        "FACE_DETECTION_MODEL",
        "FACE_LANDMARKS",
        "FILE_3D",
        "FILE_3D_FBX",
        "FILE_3D_GLB",
        "FILE_3D_GLTF",
        "FILE_3D_KSPLAT",
        "FILE_3D_OBJ",
        "FILE_3D_PLY",
        "FILE_3D_POINT_CLOUD_ANY",
        "FILE_3D_SPLAT",
        "FILE_3D_SPLAT_ANY",
        "FILE_3D_SPZ",
        "FILE_3D_STL",
        "FILE_3D_USDZ",
        "FLOAT",
        "FLOATS",
        "INTERP_MODEL",
        "GEMINI_INPUT_FILES",
        "GET_FILENAME_LIST",
        "GLIGEN",
        "GUIDER",
        "HISTOGRAM",
        "HOOKS",
        "HOOK_KEYFRAMES",
        "IC_LORA_PARAMETERS",
        "IMAGE",
        "IMAGECOMPARE",
        "INT",
        "LATENT",
        "LATENT_OPERATION",
        "LATENT_UPSCALE_MODEL",
        "LOAD3D_CAMERA",
        "LOAD3D_MODEL_INFO",
        "LOAD_3D",
        "LORA_MODEL",
        "LOSS_MAP",
        "MASK",
        "MESH",
        "MESHY_RIGGED_TASK_ID",
        "MESHY_TASK_ID",
        "MODEL",
        "MODEL_PATCH",
        "MODEL_TASK_ID",
        "MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID",
        "MOGE_GEOMETRY",
        "MOGE_MODEL",
        "NOISE",
        "OPENAI_CHAT_CONFIG",
        "OPENAI_INPUT_FILES",
        "OPTICAL_FLOW",
        "PATHS",
        "PHOTOMAKER",
        "POSE_KEYPOINT",
        "PROMPT",
        "RETARGET_TASK_ID",
        "RIG_TASK_ID",
        "SAM3_TRACK_DATA",
        "SAMPLER",
        "SAMPLERS",
        "SCHEDULERS",
        "SIGMAS",
        "SORTED",
        "SPLAT",
        "STRING",
        "STYLE_MODEL",
        "SVG",
        "TIMESTEPS_RANGE",
        "TRACKS",
        "UPSCALE_METHODS",
        "UPSCALE_MODEL",
        "VAE",
        "VAE_LIST",
        "VIDEO",
        "VOXEL",
        "WAN_CAMERA_EMBEDDING",
        "WEBCAM",
        "_COLOR_CHANNELS",
        "_LIST",
    ];

    const CUSTOM_SOURCE_IDENTITIES: &[&str] = &[
        "ELEVENLABS_VOICE",
        "KreaIO.STYLE_REF",
        "LumaIO.LUMA_CONCEPTS",
        "LumaIO.LUMA_REF",
        "LumaIO.LUMA_RAY32_KEYFRAME",
        "PixverseIO.TEMPLATE",
        "RecraftIO.COLOR",
        "RecraftIO.CONTROLS",
        "RecraftIO.STYLEV3",
        "RunwayAleph2IO.KEYFRAME",
        "RunwayAleph2IO.PROMPT_IMAGE",
    ];

    fn portable_source_types(registry: &NodeRegistry) -> Result<BTreeSet<String>, Box<dyn Error>> {
        let mut source_types = BTreeSet::new();
        for identifier in registry.registered().keys() {
            let schema = registry
                .source_schema(identifier)
                .ok_or("registered node source schema is absent")?;
            for input in &schema.inputs {
                source_types.extend(input.schema.source_type_names.iter().cloned());
            }
            for dynamic in &schema.dynamic_inputs {
                source_types.extend(dynamic.input.source_type_names.iter().cloned());
            }
            source_types.extend(
                schema
                    .outputs
                    .iter()
                    .map(|output| output.source_type_name.clone()),
            );
        }
        Ok(source_types)
    }

    #[test]
    fn source_type_projection_is_exhaustive_for_the_checked_catalog() -> Result<(), Box<dyn Error>>
    {
        let registry = NodeRegistry::built_in()?;
        let source_types = portable_source_types(&registry)?;
        let expected = PORTABLE_SOURCE_TYPES
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(PORTABLE_SOURCE_TYPES.len(), 108);
        assert_eq!(source_types, expected);
        assert!(!source_types.contains("MULTITYPE"));

        let mut owners = [0_usize; 4];
        for source_type in &source_types {
            if source_type == "CUSTOM" {
                assert!(matches!(
                    native_source_type_projection(source_type),
                    Err(NativeSourceTypeError::SourceIdentityRequired(_))
                ));
                owners[3] += 1;
                continue;
            }
            let projection = native_source_type_projection(source_type)?;
            if let Some(handle_type) = projection.handle_type()? {
                handle_type.validate()?;
            }
            owners[match projection.owner() {
                NativeSourceTypeOwner::Inline => 0,
                NativeSourceTypeOwner::Compute => 1,
                NativeSourceTypeOwner::MediaAsset => 2,
                NativeSourceTypeOwner::Provider => 3,
            }] += 1;
        }
        assert_eq!(owners, [31, 38, 29, 10]);
        Ok(())
    }

    #[test]
    fn every_checked_catalog_port_resolves_without_a_fallback() -> Result<(), Box<dyn Error>> {
        let registry = NodeRegistry::built_in()?;
        assert_eq!(registry.registered().len(), 789);
        let mut input_count = 0_usize;
        let mut output_count = 0_usize;
        for identifier in registry.registered().keys() {
            let schema = registry
                .source_schema(identifier)
                .ok_or("registered node source schema is absent")?;
            assert!(schema.unresolved_inputs.is_empty(), "{identifier}");
            assert!(schema.unresolved_outputs.is_empty(), "{identifier}");
            for input in &schema.inputs {
                native_value_types_for_input_schema(&input.schema)?;
                input_count = input_count.checked_add(1).ok_or("input count overflowed")?;
            }
            for dynamic in &schema.dynamic_inputs {
                native_value_types_for_input_schema(&dynamic.input)?;
                input_count = input_count.checked_add(1).ok_or("input count overflowed")?;
            }
            for output in &schema.outputs {
                native_value_type_for_output_schema(&NativeOutputSchemaMetadata {
                    name: output
                        .source_name
                        .clone()
                        .unwrap_or_else(|| format!("output_{}", output.ordinal)),
                    source_type_name: output.source_type_name.clone(),
                    display_name: output.display_name.clone(),
                    tooltip: output.tooltip.clone(),
                    choices: output.choices.clone(),
                    match_template: output.match_template.clone(),
                    extra: output.extra.clone(),
                })?;
                output_count = output_count
                    .checked_add(1)
                    .ok_or("output count overflowed")?;
            }
        }
        assert_eq!(input_count, 3_417);
        assert_eq!(output_count, 1_044);
        Ok(())
    }

    #[test]
    fn source_type_projection_is_case_sensitive_and_has_no_artifact_fallback()
    -> Result<(), Box<dyn Error>> {
        for source_type in PORTABLE_SOURCE_TYPES {
            let lowercase = source_type.to_ascii_lowercase();
            if lowercase != *source_type {
                assert!(matches!(
                    native_source_type_projection(&lowercase),
                    Err(NativeSourceTypeError::Unknown(_))
                ));
            }
        }
        for invented in [
            "CLIPVISION",
            "CLIPVISIONOUTPUT",
            "CONTROLNET",
            "MODELPATCH",
            "MULTITYPE",
        ] {
            assert!(matches!(
                native_source_type_projection(invented),
                Err(NativeSourceTypeError::Unknown(_))
            ));
        }
        assert!(matches!(
            native_source_type_projection("UNREGISTERED"),
            Err(NativeSourceTypeError::Unknown(_))
        ));
        let tripo = native_source_type_projection("MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID")?;
        assert_eq!(
            tripo.handle_type()?.ok_or("missing handle type")?.type_id,
            "PROVIDER_TASK_TRIPO_MODEL_RIG_RETARGET"
        );
        Ok(())
    }

    #[test]
    fn custom_source_identities_are_expression_keyed() -> Result<(), Box<dyn Error>> {
        let projected = CUSTOM_SOURCE_IDENTITIES
            .iter()
            .copied()
            .map(native_custom_source_type_projection)
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            projected
                .iter()
                .map(NativeSourceTypeProjection::role)
                .collect::<BTreeSet<_>>()
                .len(),
            11
        );
        for (identity, projection) in CUSTOM_SOURCE_IDENTITIES.iter().zip(&projected) {
            projection
                .handle_type()?
                .ok_or("custom projection is not handle-backed")?
                .validate()?;
            assert!(matches!(
                native_custom_source_type_projection(&identity.to_ascii_lowercase()),
                Err(NativeSourceTypeError::Unknown(_))
            ));
        }
        assert!(matches!(
            native_source_type_projection("CUSTOM"),
            Err(NativeSourceTypeError::SourceIdentityRequired(_))
        ));
        Ok(())
    }

    #[test]
    fn schema_scalar_and_custom_types_require_exact_port_metadata() -> Result<(), Box<dyn Error>> {
        assert!(matches!(
            native_source_type_projection("COMBO")?.value_type(),
            Err(NativeSourceTypeError::SchemaRequired(source_type)) if source_type == "COMBO"
        ));

        let empty_combo = NativeInputSchemaMetadata::compatibility("choice", "COMBO");
        let empty_combo_types = native_value_types_for_input_schema(&empty_combo)?;
        assert_eq!(
            empty_combo_types.members(),
            &[NativeValueType::NamedPreservedUnknown("COMBO".to_owned())]
        );
        assert!(
            empty_combo_types.accepts(&crate::NativeValue::PreservedUnknown {
                type_name: "COMBO".to_owned(),
                value: serde_json::json!("source-defined-dynamic-choice"),
            })
        );
        assert!(!empty_combo_types.accepts(&crate::NativeValue::Primitive {
            value: crate::NativePrimitive::String("source-defined-dynamic-choice".to_owned()),
        }));

        let mut combo = NativeInputSchemaMetadata::compatibility("choice", "COMBO");
        combo.choices = vec![
            NativeSchemaValue::String {
                value: "one".to_owned(),
            },
            NativeSchemaValue::UnsignedInteger { value: 2 },
        ];
        assert_eq!(
            native_value_types_for_input_schema(&combo)?.members(),
            &[
                NativeValueType::Primitive(NativePrimitiveType::Integer),
                NativeValueType::Primitive(NativePrimitiveType::String),
            ]
        );

        let dynamic = NativeInputSchemaMetadata::compatibility("choice", "COMFY_DYNAMICCOMBO_V3");
        assert_eq!(
            native_value_types_for_input_schema(&dynamic)?.members(),
            &[NativeValueType::NamedPreservedUnknown(
                "COMFY_DYNAMICCOMBO_V3".to_owned()
            )]
        );

        let curve = native_source_type_projection("CURVE")?.value_type()?;
        let histogram = native_source_type_projection("HISTOGRAM")?.value_type()?;
        assert_ne!(curve, histogram);
        let curve_union = NativeTypeUnion::new([curve])?;
        assert!(curve_union.accepts(&crate::NativeValue::PreservedUnknown {
            type_name: "CURVE".to_owned(),
            value: serde_json::json!({"points": []}),
        }));
        assert!(!curve_union.accepts(&crate::NativeValue::PreservedUnknown {
            type_name: "HISTOGRAM".to_owned(),
            value: serde_json::json!([]),
        }));

        let mut custom = NativeInputSchemaMetadata::compatibility("voice", "CUSTOM");
        assert!(matches!(
            native_value_types_for_input_schema(&custom),
            Err(NativeSourceTypeError::SourceIdentityRequired(_))
        ));
        custom.extra.push(NativeSchemaField {
            name: "source_identity".to_owned(),
            value: NativeSchemaValue::String {
                value: "ELEVENLABS_VOICE".to_owned(),
            },
        });
        assert_eq!(
            native_value_types_for_input_schema(&custom)?.members(),
            &[NativeValueType::Handle(NativeHandleType::new(
                NativeHandleKind::ProviderTask,
                "CUSTOM_ELEVENLABS_VOICE",
            )?)]
        );
        Ok(())
    }

    #[test]
    fn file_3d_union_sockets_admit_only_their_concrete_source_formats() -> Result<(), Box<dyn Error>>
    {
        fn handle(source_type: &str) -> Result<NativeHandleType, Box<dyn Error>> {
            Ok(native_source_type_projection(source_type)?
                .handle_type()?
                .ok_or_else(|| format!("{source_type} is not handle-backed"))?)
        }
        let any = handle("FILE_3D")?;
        let point_cloud = handle("FILE_3D_POINT_CLOUD_ANY")?;
        let splat = handle("FILE_3D_SPLAT_ANY")?;
        let ply = handle("FILE_3D_PLY")?;
        let spz = handle("FILE_3D_SPZ")?;
        let glb = handle("FILE_3D_GLB")?;

        assert!(native_handle_type_accepts(&any, &ply));
        assert!(native_handle_type_accepts(&any, &glb));
        assert!(native_handle_type_accepts(&point_cloud, &ply));
        assert!(!native_handle_type_accepts(&point_cloud, &spz));
        assert!(native_handle_type_accepts(&splat, &ply));
        assert!(native_handle_type_accepts(&splat, &spz));
        assert!(!native_handle_type_accepts(&splat, &glb));
        assert!(!native_handle_type_accepts(&ply, &splat));
        Ok(())
    }
}
