use serde::{Deserialize, Serialize};
use sim_nodes::{NativeSourceTypeError, native_plugin_source_type_projection};
use std::{collections::BTreeMap, error::Error, fmt, str::FromStr};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueFamily {
    Scalar,
    Tensor,
    Artifact,
    Model,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueRepresentation {
    ScalarV1,
    TensorV1,
    ArtifactV1,
    ModelV1,
}

impl ValueRepresentation {
    pub const fn family(self) -> ValueFamily {
        match self {
            Self::ScalarV1 => ValueFamily::Scalar,
            Self::TensorV1 => ValueFamily::Tensor,
            Self::ArtifactV1 => ValueFamily::Artifact,
            Self::ModelV1 => ValueFamily::Model,
        }
    }

    pub const fn wire_schema(self) -> &'static str {
        match self {
            Self::ScalarV1 => "zed:sim-value/scalar@1",
            Self::TensorV1 => "zed:sim-value/tensor@1",
            Self::ArtifactV1 => "zed:sim-value/artifact@1",
            Self::ModelV1 => "zed:sim-value/model@1",
        }
    }
}

impl ValueFamily {
    pub const fn representation(self) -> ValueRepresentation {
        match self {
            Self::Scalar => ValueRepresentation::ScalarV1,
            Self::Tensor => ValueRepresentation::TensorV1,
            Self::Artifact => ValueRepresentation::ArtifactV1,
            Self::Model => ValueRepresentation::ModelV1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TypeEvolutionRule {
    AdditiveWithinMajor,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CanonicalTypeId {
    namespace: String,
    name: String,
    major: u16,
}

impl CanonicalTypeId {
    pub fn new(
        namespace: impl Into<String>,
        name: impl Into<String>,
        major: u16,
    ) -> Result<Self, TypeRegistryError> {
        let identifier = Self {
            namespace: namespace.into(),
            name: name.into(),
            major,
        };
        identifier.validate()?;
        Ok(identifier)
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn major(&self) -> u16 {
        self.major
    }

    fn validate(&self) -> Result<(), TypeRegistryError> {
        if !valid_identifier_segment(&self.namespace) || !valid_identifier_segment(&self.name) {
            return Err(TypeRegistryError::InvalidTypeId(self.to_string()));
        }
        if self.major == 0 {
            return Err(TypeRegistryError::InvalidTypeId(self.to_string()));
        }
        Ok(())
    }
}

impl fmt::Display for CanonicalTypeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}@{}", self.namespace, self.name, self.major)
    }
}

impl FromStr for CanonicalTypeId {
    type Err = TypeRegistryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (qualified_name, major) = value
            .rsplit_once('@')
            .ok_or_else(|| TypeRegistryError::InvalidTypeId(value.to_owned()))?;
        let (namespace, name) = qualified_name
            .split_once(':')
            .ok_or_else(|| TypeRegistryError::InvalidTypeId(value.to_owned()))?;
        if name.contains(':') {
            return Err(TypeRegistryError::InvalidTypeId(value.to_owned()));
        }
        let major = major
            .parse::<u16>()
            .map_err(|_| TypeRegistryError::InvalidTypeId(value.to_owned()))?;
        Self::new(namespace, name, major)
    }
}

impl TryFrom<String> for CanonicalTypeId {
    type Error = TypeRegistryError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl From<CanonicalTypeId> for String {
    fn from(value: CanonicalTypeId) -> Self {
        value.to_string()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypeRegistration {
    pub canonical: &'static str,
    pub source_socket: &'static str,
    pub publisher: &'static str,
    pub family: ValueFamily,
    pub representation: ValueRepresentation,
    pub wire_schema: &'static str,
    pub evolution: TypeEvolutionRule,
    pub aliases: &'static [&'static str],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TypeRegistryError {
    InvalidTypeId(String),
    DuplicateCanonical(String),
    DuplicateSourceSocket(String),
    AliasCollision(String),
    InvalidPublisher(String),
    NamespaceCollision {
        namespace: String,
        expected_publisher: String,
        actual_publisher: String,
    },
    SchemaCollision {
        type_id: String,
        expected_schema: String,
        actual_schema: String,
    },
    UnknownType(String),
    FamilyMismatch {
        type_id: String,
        expected: ValueFamily,
        actual: ValueFamily,
    },
    RepresentationMismatch {
        type_id: String,
        expected: ValueRepresentation,
        actual: ValueRepresentation,
    },
    SourceProjection {
        type_id: String,
        source_socket: String,
        reason: String,
    },
}

impl fmt::Display for TypeRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTypeId(value) => write!(formatter, "invalid canonical type ID `{value}`"),
            Self::DuplicateCanonical(value) => {
                write!(formatter, "duplicate canonical type ID `{value}`")
            }
            Self::DuplicateSourceSocket(value) => {
                write!(formatter, "duplicate source socket `{value}`")
            }
            Self::AliasCollision(value) => write!(formatter, "type alias collision for `{value}`"),
            Self::InvalidPublisher(value) => {
                write!(formatter, "invalid type publisher identifier `{value}`")
            }
            Self::NamespaceCollision {
                namespace,
                expected_publisher,
                actual_publisher,
            } => write!(
                formatter,
                "namespace `{namespace}` belongs to `{expected_publisher}`, not `{actual_publisher}`"
            ),
            Self::SchemaCollision {
                type_id,
                expected_schema,
                actual_schema,
            } => write!(
                formatter,
                "type `{type_id}` uses wire schema `{actual_schema}`, not `{expected_schema}`"
            ),
            Self::UnknownType(value) => write!(formatter, "unknown plugin type `{value}`"),
            Self::FamilyMismatch {
                type_id,
                expected,
                actual,
            } => write!(
                formatter,
                "type `{type_id}` belongs to {actual:?}, not expected {expected:?}"
            ),
            Self::RepresentationMismatch {
                type_id,
                expected,
                actual,
            } => write!(
                formatter,
                "type `{type_id}` uses {actual:?}, not expected {expected:?}"
            ),
            Self::SourceProjection {
                type_id,
                source_socket,
                reason,
            } => write!(
                formatter,
                "type `{type_id}` source socket `{source_socket}` is not canonical: {reason}"
            ),
        }
    }
}

impl Error for TypeRegistryError {}

#[derive(Clone, Debug)]
pub struct TypeRegistry {
    canonical: BTreeMap<CanonicalTypeId, RegisteredType>,
    aliases: BTreeMap<String, CanonicalTypeId>,
}

#[derive(Clone, Debug)]
struct RegisteredType {
    publisher: String,
    representation: ValueRepresentation,
    wire_schema: String,
    evolution: TypeEvolutionRule,
}

impl TypeRegistry {
    pub fn built_in() -> Result<Self, TypeRegistryError> {
        let registry = TypeRegistry::from_registrations(BUILT_IN_TYPES)?;
        validate_native_source_projections(BUILT_IN_TYPES)?;
        Ok(registry)
    }

    pub fn from_registrations(
        registrations: &[TypeRegistration],
    ) -> Result<Self, TypeRegistryError> {
        let mut canonical = BTreeMap::new();
        let mut aliases = BTreeMap::new();
        let mut source_sockets = BTreeMap::new();
        let mut namespace_publishers = BTreeMap::<String, String>::new();
        for registration in registrations {
            if !valid_publisher(registration.publisher) {
                return Err(TypeRegistryError::InvalidPublisher(
                    registration.publisher.to_owned(),
                ));
            }
            if registration.representation.family() != registration.family {
                return Err(TypeRegistryError::RepresentationMismatch {
                    type_id: registration.canonical.to_owned(),
                    expected: registration.family.representation(),
                    actual: registration.representation,
                });
            }
            let type_id = registration.canonical.parse::<CanonicalTypeId>()?;
            if let Some(expected_publisher) = namespace_publishers.get(type_id.namespace()) {
                if expected_publisher != registration.publisher {
                    return Err(TypeRegistryError::NamespaceCollision {
                        namespace: type_id.namespace().to_owned(),
                        expected_publisher: expected_publisher.clone(),
                        actual_publisher: registration.publisher.to_owned(),
                    });
                }
            } else {
                namespace_publishers.insert(
                    type_id.namespace().to_owned(),
                    registration.publisher.to_owned(),
                );
            }
            let expected_schema = registration.representation.wire_schema();
            if registration.wire_schema != expected_schema {
                return Err(TypeRegistryError::SchemaCollision {
                    type_id: registration.canonical.to_owned(),
                    expected_schema: expected_schema.to_owned(),
                    actual_schema: registration.wire_schema.to_owned(),
                });
            }
            if canonical
                .insert(
                    type_id.clone(),
                    RegisteredType {
                        publisher: registration.publisher.to_owned(),
                        representation: registration.representation,
                        wire_schema: registration.wire_schema.to_owned(),
                        evolution: registration.evolution,
                    },
                )
                .is_some()
            {
                return Err(TypeRegistryError::DuplicateCanonical(
                    registration.canonical.to_owned(),
                ));
            }
            if source_sockets
                .insert(registration.source_socket, registration.canonical)
                .is_some()
            {
                return Err(TypeRegistryError::DuplicateSourceSocket(
                    registration.source_socket.to_owned(),
                ));
            }
            insert_alias(&mut aliases, registration.source_socket, &type_id)?;
            insert_alias(&mut aliases, registration.canonical, &type_id)?;
            for alias in registration.aliases {
                insert_alias(&mut aliases, alias, &type_id)?;
            }
        }
        Ok(Self { canonical, aliases })
    }

    pub fn resolve(
        &self,
        identifier_or_alias: &str,
    ) -> Result<&CanonicalTypeId, TypeRegistryError> {
        self.aliases
            .get(identifier_or_alias)
            .ok_or_else(|| TypeRegistryError::UnknownType(identifier_or_alias.to_owned()))
    }

    pub fn family(&self, type_id: &CanonicalTypeId) -> Result<ValueFamily, TypeRegistryError> {
        self.canonical
            .get(type_id)
            .map(|registration| registration.representation.family())
            .ok_or_else(|| TypeRegistryError::UnknownType(type_id.to_string()))
    }

    pub fn representation(
        &self,
        type_id: &CanonicalTypeId,
    ) -> Result<ValueRepresentation, TypeRegistryError> {
        self.canonical
            .get(type_id)
            .map(|registration| registration.representation)
            .ok_or_else(|| TypeRegistryError::UnknownType(type_id.to_string()))
    }

    pub fn publisher(&self, type_id: &CanonicalTypeId) -> Result<&str, TypeRegistryError> {
        self.canonical
            .get(type_id)
            .map(|registration| registration.publisher.as_str())
            .ok_or_else(|| TypeRegistryError::UnknownType(type_id.to_string()))
    }

    pub fn wire_schema(&self, type_id: &CanonicalTypeId) -> Result<&str, TypeRegistryError> {
        self.canonical
            .get(type_id)
            .map(|registration| registration.wire_schema.as_str())
            .ok_or_else(|| TypeRegistryError::UnknownType(type_id.to_string()))
    }

    pub fn evolution(
        &self,
        type_id: &CanonicalTypeId,
    ) -> Result<TypeEvolutionRule, TypeRegistryError> {
        self.canonical
            .get(type_id)
            .map(|registration| registration.evolution)
            .ok_or_else(|| TypeRegistryError::UnknownType(type_id.to_string()))
    }

    pub fn require_family(
        &self,
        type_id: &CanonicalTypeId,
        expected: ValueFamily,
    ) -> Result<(), TypeRegistryError> {
        let actual = self.family(type_id)?;
        if actual != expected {
            return Err(TypeRegistryError::FamilyMismatch {
                type_id: type_id.to_string(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.canonical.len()
    }

    pub fn is_empty(&self) -> bool {
        self.canonical.is_empty()
    }

    pub fn registrations(&self) -> impl Iterator<Item = (&CanonicalTypeId, ValueFamily)> {
        self.canonical
            .iter()
            .map(|(identifier, registration)| (identifier, registration.representation.family()))
    }

    pub fn canonical_projection(&self) -> String {
        let mut projection = String::new();
        for (identifier, registration) in &self.canonical {
            projection.push_str("type|");
            projection.push_str(&identifier.to_string());
            projection.push('|');
            projection.push_str(&registration.publisher);
            projection.push('|');
            projection.push_str(match registration.representation {
                ValueRepresentation::ScalarV1 => "scalar-v1",
                ValueRepresentation::TensorV1 => "tensor-v1",
                ValueRepresentation::ArtifactV1 => "artifact-v1",
                ValueRepresentation::ModelV1 => "model-v1",
            });
            projection.push('|');
            projection.push_str(&registration.wire_schema);
            projection.push('|');
            projection.push_str(match registration.evolution {
                TypeEvolutionRule::AdditiveWithinMajor => "additive-within-major",
            });
            projection.push('\n');
        }
        for (alias, identifier) in &self.aliases {
            projection.push_str("alias|");
            projection.push_str(alias);
            projection.push('|');
            projection.push_str(&identifier.to_string());
            projection.push('\n');
        }
        projection
    }
}

fn validate_native_source_projections(
    registrations: &[TypeRegistration],
) -> Result<(), TypeRegistryError> {
    for registration in registrations {
        let type_id = registration.canonical.parse::<CanonicalTypeId>()?;
        let projection = match native_plugin_source_type_projection(type_id.name()) {
            Ok(projection) => projection,
            Err(NativeSourceTypeError::SourceIdentityRequired(_))
                if matches!(type_id.name(), "autogrow" | "custom" | "multi-type") =>
            {
                continue;
            }
            Err(error) => {
                return Err(TypeRegistryError::SourceProjection {
                    type_id: registration.canonical.to_owned(),
                    source_socket: registration.source_socket.to_owned(),
                    reason: error.to_string(),
                });
            }
        };
        let source_type = projection.source_type();
        let custom_identity = type_id.name().starts_with("custom-") && source_type == "CUSTOM";
        if !custom_identity
            && registration.source_socket != source_type
            && !registration.aliases.contains(&source_type)
        {
            return Err(TypeRegistryError::SourceProjection {
                type_id: registration.canonical.to_owned(),
                source_socket: registration.source_socket.to_owned(),
                reason: format!("authoritative source identity is `{source_type}`"),
            });
        }
    }
    Ok(())
}

fn insert_alias(
    aliases: &mut BTreeMap<String, CanonicalTypeId>,
    alias: &str,
    type_id: &CanonicalTypeId,
) -> Result<(), TypeRegistryError> {
    if aliases.insert(alias.to_owned(), type_id.clone()).is_some() {
        return Err(TypeRegistryError::AliasCollision(alias.to_owned()));
    }
    Ok(())
}

fn valid_identifier_segment(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && value.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}

fn valid_publisher(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && value.split('.').all(valid_identifier_segment)
}

macro_rules! plugin_types {
    ($(($canonical:literal, $source:literal, $family:ident, [$($alias:literal),* $(,)?])),* $(,)?) => {
        pub const BUILT_IN_TYPES: &[TypeRegistration] = &[
            $(TypeRegistration {
                canonical: $canonical,
                source_socket: $source,
                publisher: "zed",
                family: ValueFamily::$family,
                representation: ValueFamily::$family.representation(),
                wire_schema: ValueFamily::$family.representation().wire_schema(),
                evolution: TypeEvolutionRule::AdditiveWithinMajor,
                aliases: &[$($alias),*],
            }),*
        ];
    };
}

plugin_types!(
    ("sim:any@1", "AnyType", Scalar, ["*"]),
    ("sim:array@1", "Array", Scalar, ["ARRAY"]),
    ("sim:audio@1", "Audio", Tensor, ["AUDIO"]),
    (
        "sim:audio-encoder@1",
        "AudioEncoder",
        Model,
        ["AUDIO_ENCODER"]
    ),
    (
        "sim:audio-encoder-output@1",
        "AudioEncoderOutput",
        Tensor,
        ["AUDIO_ENCODER_OUTPUT"]
    ),
    ("sim:autogrow@1", "Autogrow", Scalar, []),
    (
        "sim:background-removal@1",
        "BackgroundRemoval",
        Model,
        ["BACKGROUND_REMOVAL"]
    ),
    ("sim:boolean@1", "Boolean", Scalar, ["BOOLEAN"]),
    (
        "sim:bounding-box@1",
        "BoundingBox",
        Scalar,
        ["BOUNDING_BOX"]
    ),
    (
        "sim:bounding-box-editor@1",
        "BoundingBoxes",
        Scalar,
        ["BOUNDING_BOXES"]
    ),
    ("sim:clip@1", "Clip", Model, ["CLIP"]),
    ("sim:clip-vision@1", "ClipVision", Model, ["CLIP_VISION"]),
    (
        "sim:clip-vision-output@1",
        "ClipVisionOutput",
        Tensor,
        ["CLIP_VISION_OUTPUT"]
    ),
    ("sim:color@1", "Color", Scalar, ["COLOR"]),
    ("sim:color-list@1", "Colors", Scalar, ["COLORS"]),
    ("sim:combo@1", "Combo", Scalar, ["COMBO"]),
    (
        "sim:conditioning@1",
        "Conditioning",
        Tensor,
        ["CONDITIONING"]
    ),
    ("sim:control-net@1", "ControlNet", Model, ["CONTROL_NET"]),
    ("sim:curve@1", "Curve", Tensor, ["CURVE"]),
    ("sim:custom@1", "Custom", Scalar, ["CUSTOM"]),
    ("sim:dictionary@1", "Dict", Scalar, ["DICT"]),
    (
        "sim:dynamic-combo@1",
        "DynamicCombo",
        Scalar,
        ["DYNAMIC_COMBO", "COMFY_DYNAMICCOMBO_V3"]
    ),
    (
        "sim:file-3d-any@1",
        "File3DAny",
        Artifact,
        ["FILE_3D", "FILE_3D_ANY"]
    ),
    ("sim:file-3d-fbx@1", "File3DFBX", Artifact, ["FILE_3D_FBX"]),
    ("sim:file-3d-glb@1", "File3DGLB", Artifact, ["FILE_3D_GLB"]),
    ("sim:file-3d-obj@1", "File3DOBJ", Artifact, ["FILE_3D_OBJ"]),
    (
        "sim:file-3d-point-cloud@1",
        "File3DPointCloudAny",
        Artifact,
        ["FILE_3D_POINT_CLOUD_ANY"]
    ),
    (
        "sim:file-3d-splat@1",
        "File3DSplatAny",
        Artifact,
        ["FILE_3D_SPLAT_ANY"]
    ),
    ("sim:float@1", "Float", Scalar, ["FLOAT"]),
    ("sim:float-list@1", "FloatList", Scalar, ["FLOATS"]),
    ("sim:gligen@1", "Gligen", Model, ["GLIGEN"]),
    ("sim:guider@1", "Guider", Model, ["GUIDER"]),
    ("sim:histogram@1", "Histogram", Tensor, ["HISTOGRAM"]),
    ("sim:hooks@1", "Hooks", Model, ["HOOKS"]),
    (
        "sim:hook-keyframes@1",
        "HookKeyframes",
        Model,
        ["HOOK_KEYFRAMES", "HOOK_KF"]
    ),
    ("sim:image@1", "Image", Tensor, ["IMAGE"]),
    (
        "sim:image-compare@1",
        "ImageCompare",
        Scalar,
        ["IMAGECOMPARE", "IMAGE_COMPARE"]
    ),
    ("sim:integer@1", "Int", Scalar, ["INT"]),
    ("sim:latent@1", "Latent", Tensor, ["LATENT"]),
    (
        "sim:latent-operation@1",
        "LatentOperation",
        Model,
        ["LATENT_OPERATION"]
    ),
    (
        "sim:latent-upscale-model@1",
        "LatentUpscaleModel",
        Model,
        ["LATENT_UPSCALE_MODEL"]
    ),
    ("sim:load-3d@1", "Load3D", Artifact, ["LOAD_3D"]),
    (
        "sim:load-3d-camera@1",
        "Load3DCamera",
        Scalar,
        ["LOAD3D_CAMERA", "LOAD_3D_CAMERA"]
    ),
    (
        "sim:load-3d-model-info@1",
        "Load3DModelInfo",
        Scalar,
        ["LOAD3D_MODEL_INFO", "LOAD_3D_MODEL_INFO"]
    ),
    ("sim:mask@1", "Mask", Tensor, ["MASK"]),
    (
        "sim:match-type@1",
        "MatchType",
        Scalar,
        ["COMFY_MATCHTYPE_V3", "MATCH_TYPE"]
    ),
    ("sim:mesh@1", "Mesh", Artifact, ["MESH"]),
    ("sim:model@1", "Model", Model, ["MODEL"]),
    ("sim:model-patch@1", "ModelPatch", Model, ["MODEL_PATCH"]),
    ("sim:multi-type@1", "MultiType", Scalar, ["MULTI_TYPE"]),
    ("sim:noise@1", "Noise", Tensor, ["NOISE"]),
    ("sim:photomaker@1", "Photomaker", Model, ["PHOTOMAKER"]),
    ("sim:sampler@1", "Sampler", Model, ["SAMPLER"]),
    ("sim:sigmas@1", "Sigmas", Tensor, ["SIGMAS"]),
    ("sim:splat@1", "Splat", Artifact, ["SPLAT"]),
    ("sim:string@1", "String", Scalar, ["STRING"]),
    ("sim:style-model@1", "StyleModel", Model, ["STYLE_MODEL"]),
    ("sim:svg@1", "SVG", Artifact, []),
    (
        "sim:timesteps-range@1",
        "TimestepsRange",
        Scalar,
        ["TIMESTEPS_RANGE"]
    ),
    ("sim:tracks@1", "Tracks", Tensor, ["TRACKS"]),
    (
        "sim:upscale-model@1",
        "UpscaleModel",
        Model,
        ["UPSCALE_MODEL"]
    ),
    ("sim:vae@1", "Vae", Model, ["VAE"]),
    ("sim:video@1", "Video", Tensor, ["VIDEO"]),
    ("sim:voxel@1", "Voxel", Tensor, ["VOXEL"]),
    (
        "sim:wan-camera-embedding@1",
        "WanCameraEmbedding",
        Tensor,
        ["WAN_CAMERA_EMBEDDING"]
    ),
    ("sim:webcam@1", "Webcam", Tensor, ["WEBCAM"]),
    ("sim:camera-control@1", "CAMERA_CONTROL", Artifact, []),
    ("sim:gemini-input-files@1", "GEMINI_INPUT_FILES", Model, []),
    (
        "sim:meshy-rigged-task-id@1",
        "MESHY_RIGGED_TASK_ID",
        Model,
        []
    ),
    ("sim:meshy-task-id@1", "MESHY_TASK_ID", Model, []),
    ("sim:model-task-id@1", "MODEL_TASK_ID", Model, []),
    ("sim:openai-chat-config@1", "OPENAI_CHAT_CONFIG", Model, []),
    ("sim:openai-input-files@1", "OPENAI_INPUT_FILES", Model, []),
    ("sim:retarget-task-id@1", "RETARGET_TASK_ID", Model, []),
    ("sim:rig-task-id@1", "RIG_TASK_ID", Model, []),
    (
        "sim:custom-elevenlabs-voice@1",
        "ELEVENLABS_VOICE",
        Model,
        []
    ),
    ("sim:custom-krea-style-ref@1", "KreaIO.STYLE_REF", Model, []),
    (
        "sim:custom-luma-concepts@1",
        "LumaIO.LUMA_CONCEPTS",
        Model,
        []
    ),
    (
        "sim:custom-luma-ray32-keyframe@1",
        "LumaIO.LUMA_RAY32_KEYFRAME",
        Model,
        []
    ),
    ("sim:custom-luma-ref@1", "LumaIO.LUMA_REF", Model, []),
    (
        "sim:custom-pixverse-template@1",
        "PixverseIO.TEMPLATE",
        Model,
        []
    ),
    ("sim:custom-recraft-color@1", "RecraftIO.COLOR", Model, []),
    (
        "sim:custom-recraft-controls@1",
        "RecraftIO.CONTROLS",
        Model,
        []
    ),
    (
        "sim:custom-recraft-style-v3@1",
        "RecraftIO.STYLEV3",
        Model,
        []
    ),
    (
        "sim:custom-runway-aleph2-keyframe@1",
        "RunwayAleph2IO.KEYFRAME",
        Model,
        []
    ),
    (
        "sim:custom-runway-aleph2-prompt-image@1",
        "RunwayAleph2IO.PROMPT_IMAGE",
        Model,
        []
    ),
);

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;

    #[test]
    fn canonical_ids_round_trip() -> Result<(), Box<dyn Error>> {
        let type_id = "sim:image@1".parse::<CanonicalTypeId>()?;
        assert_eq!(type_id.namespace(), "sim");
        assert_eq!(type_id.name(), "image");
        assert_eq!(type_id.major(), 1);
        assert_eq!(type_id.to_string(), "sim:image@1");
        for invalid in ["image", "sim:image", "Sim:image@1", "sim:image@0"] {
            assert!(invalid.parse::<CanonicalTypeId>().is_err());
        }
        Ok(())
    }

    #[test]
    fn built_in_registry_has_one_canonical_entry_per_socket() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        assert_eq!(registry.len(), 86);
        assert!(registry.resolve("comfy:image@1").is_err());
        assert!(registry.resolve("comfy:string@1").is_err());
        assert_eq!(
            registry.resolve("CLIP_VISION")?.to_string(),
            "sim:clip-vision@1"
        );
        assert_eq!(
            registry.resolve("HOOK_KF")?.to_string(),
            "sim:hook-keyframes@1"
        );
        assert_eq!(
            registry.family(registry.resolve("IMAGE")?)?,
            ValueFamily::Tensor
        );
        assert_eq!(
            registry.evolution(registry.resolve("IMAGE")?)?,
            TypeEvolutionRule::AdditiveWithinMajor
        );
        assert_eq!(registry.publisher(registry.resolve("IMAGE")?)?, "zed");
        assert_eq!(
            registry.wire_schema(registry.resolve("IMAGE")?)?,
            "zed:sim-value/tensor@1"
        );
        assert_eq!(
            registry.resolve("MODEL_TASK_ID")?.to_string(),
            "sim:model-task-id@1"
        );
        assert_eq!(
            registry.resolve("KreaIO.STYLE_REF")?.to_string(),
            "sim:custom-krea-style-ref@1"
        );
        Ok(())
    }

    #[test]
    fn registry_covers_the_cataloged_v1_and_v3_socket_contracts_exactly()
    -> Result<(), Box<dyn Error>> {
        let catalog =
            include_str!("../../../.agents/specs/comfy-parity/catalogs/backend-nodes.csv");
        let mut catalog_sockets = BTreeSet::new();
        for marker in ["IO.", "io."] {
            let mut remainder = catalog;
            while let Some(position) = remainder.find(marker) {
                remainder = remainder
                    .get(position + marker.len()..)
                    .ok_or("catalog marker offset overflow")?;
                let identifier_length = remainder
                    .bytes()
                    .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                    .count();
                let identifier = remainder
                    .get(..identifier_length)
                    .ok_or("catalog identifier offset overflow")?;
                let suffix = remainder
                    .get(identifier_length..)
                    .ok_or("catalog suffix offset overflow")?;
                if suffix.starts_with(".Input") || suffix.starts_with(".Output") {
                    catalog_sockets.insert(identifier.to_owned());
                }
                remainder = suffix;
            }
        }
        assert_eq!(catalog_sockets.len(), 59);

        let legacy_v1_sockets = [
            "BOOLEAN",
            "CLIP",
            "CLIP_VISION",
            "CLIP_VISION_OUTPUT",
            "COMBO",
            "CONDITIONING",
            "CONTROL_NET",
            "FLOAT",
            "FLOATS",
            "GLIGEN",
            "HOOKS",
            "HOOK_KEYFRAMES",
            "HOOK_KF",
            "IMAGE",
            "INT",
            "LATENT",
            "MASK",
            "MODEL",
            "MODEL_PATCH",
            "STRING",
            "STYLE_MODEL",
            "TIMESTEPS_RANGE",
            "VAE",
            "WEBCAM",
        ];
        let registry = TypeRegistry::built_in()?;
        let mut resolved = BTreeSet::new();
        for socket in catalog_sockets
            .iter()
            .map(String::as_str)
            .chain(legacy_v1_sockets)
        {
            resolved.insert(registry.resolve(socket)?.clone());
        }
        for socket in [
            "CAMERA_CONTROL",
            "GEMINI_INPUT_FILES",
            "MESHY_RIGGED_TASK_ID",
            "MESHY_TASK_ID",
            "MODEL_TASK_ID",
            "OPENAI_CHAT_CONFIG",
            "OPENAI_INPUT_FILES",
            "RETARGET_TASK_ID",
            "RIG_TASK_ID",
            "ELEVENLABS_VOICE",
            "KreaIO.STYLE_REF",
            "LumaIO.LUMA_CONCEPTS",
            "LumaIO.LUMA_RAY32_KEYFRAME",
            "LumaIO.LUMA_REF",
            "PixverseIO.TEMPLATE",
            "RecraftIO.COLOR",
            "RecraftIO.CONTROLS",
            "RecraftIO.STYLEV3",
            "RunwayAleph2IO.KEYFRAME",
            "RunwayAleph2IO.PROMPT_IMAGE",
        ] {
            resolved.insert(registry.resolve(socket)?.clone());
        }
        assert_eq!(resolved.len(), 86);
        assert_eq!(resolved.len(), registry.len());
        assert!(
            registry
                .registrations()
                .all(|(type_id, _)| resolved.contains(type_id))
        );
        Ok(())
    }

    #[test]
    fn collisions_are_rejected() {
        let duplicate = [
            TypeRegistration {
                canonical: "sim:a@1",
                source_socket: "A",
                publisher: "zed",
                family: ValueFamily::Scalar,
                representation: ValueRepresentation::ScalarV1,
                wire_schema: ValueRepresentation::ScalarV1.wire_schema(),
                evolution: TypeEvolutionRule::AdditiveWithinMajor,
                aliases: &["SHARED"],
            },
            TypeRegistration {
                canonical: "sim:b@1",
                source_socket: "B",
                publisher: "zed",
                family: ValueFamily::Tensor,
                representation: ValueRepresentation::TensorV1,
                wire_schema: ValueRepresentation::TensorV1.wire_schema(),
                evolution: TypeEvolutionRule::AdditiveWithinMajor,
                aliases: &["SHARED"],
            },
        ];
        assert!(matches!(
            TypeRegistry::from_registrations(&duplicate),
            Err(TypeRegistryError::AliasCollision(alias)) if alias == "SHARED"
        ));
    }

    #[test]
    fn generated_registry_projection_is_stable_and_representation_checked()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let projection = registry.canonical_projection();
        assert_eq!(
            projection
                .lines()
                .filter(|line| line.starts_with("type|"))
                .count(),
            86
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(projection.as_bytes())),
            "d6da57e6dc0339dd906e09e87c924e52d9fcb92dcc6b02489b2ea8f1279618ac"
        );
        assert_eq!(
            projection,
            include_str!("../schema/type-registry-v1.projection")
        );
        assert_eq!(
            projection
                .lines()
                .filter(|line| line.starts_with("alias|"))
                .count(),
            243
        );
        assert_eq!(
            registry.representation(registry.resolve("IMAGE")?)?,
            ValueRepresentation::TensorV1
        );

        let mismatch = [TypeRegistration {
            canonical: "sim:mismatch@1",
            source_socket: "MISMATCH",
            publisher: "zed",
            family: ValueFamily::Scalar,
            representation: ValueRepresentation::TensorV1,
            wire_schema: ValueRepresentation::TensorV1.wire_schema(),
            evolution: TypeEvolutionRule::AdditiveWithinMajor,
            aliases: &[],
        }];
        assert!(matches!(
            TypeRegistry::from_registrations(&mismatch),
            Err(TypeRegistryError::RepresentationMismatch { .. })
        ));

        let namespace_collision = [
            TypeRegistration {
                canonical: "shared:a@1",
                source_socket: "SHARED_A",
                publisher: "publisher-a",
                family: ValueFamily::Scalar,
                representation: ValueRepresentation::ScalarV1,
                wire_schema: ValueRepresentation::ScalarV1.wire_schema(),
                evolution: TypeEvolutionRule::AdditiveWithinMajor,
                aliases: &[],
            },
            TypeRegistration {
                canonical: "shared:b@1",
                source_socket: "SHARED_B",
                publisher: "publisher-b",
                family: ValueFamily::Scalar,
                representation: ValueRepresentation::ScalarV1,
                wire_schema: ValueRepresentation::ScalarV1.wire_schema(),
                evolution: TypeEvolutionRule::AdditiveWithinMajor,
                aliases: &[],
            },
        ];
        assert!(matches!(
            TypeRegistry::from_registrations(&namespace_collision),
            Err(TypeRegistryError::NamespaceCollision { namespace, .. }) if namespace == "shared"
        ));

        let schema_collision = [TypeRegistration {
            canonical: "schema:value@1",
            source_socket: "SCHEMA_VALUE",
            publisher: "publisher",
            family: ValueFamily::Scalar,
            representation: ValueRepresentation::ScalarV1,
            wire_schema: "zed:sim-value/tensor@1",
            evolution: TypeEvolutionRule::AdditiveWithinMajor,
            aliases: &[],
        }];
        assert!(matches!(
            TypeRegistry::from_registrations(&schema_collision),
            Err(TypeRegistryError::SchemaCollision { .. })
        ));
        Ok(())
    }

    #[test]
    fn registry_reconciliation_consumes_every_declared_task_catalog() {
        let catalogs = [
            (
                include_bytes!("../../../.agents/specs/comfy-parity/catalogs/backend-nodes.csv")
                    .as_slice(),
                "4c966376f6b188f7bf42d00ace9ce99107708fa8d835cb68ba4a09d53d94b7c2",
                b"schema_api".as_slice(),
            ),
            (
                include_bytes!(
                    "../../../.agents/specs/comfy-parity/catalogs/cross-compatibility.csv"
                )
                .as_slice(),
                "30c6d92d5df78fc439a2aaecc7ef1a78eb0a23cd2733c1556f280e16588b6fc3",
                b"Version and capability negotiation".as_slice(),
            ),
            (
                include_bytes!(
                    "../../../.agents/specs/comfy-parity/catalogs/frontend-extensions.csv"
                )
                .as_slice(),
                "14280a8d1907a49292735692667eecff579fa5d4bf462dc8ef3287ec3bd7141b",
                b"Unique extension name".as_slice(),
            ),
            (
                include_bytes!(
                    "../../../.agents/specs/comfy-parity/catalogs/docs-extension-contracts.csv"
                )
                .as_slice(),
                "0f1f4e0b8ebb5cb2e956d2a96018b38c8e01afd340045915284a315147122f54",
                b"production_legacy_execution".as_slice(),
            ),
            (
                include_bytes!(
                    "../../../.agents/specs/comfy-parity/catalogs/backend-external-services.csv"
                )
                .as_slice(),
                "ffad144fcc049f61dea8704837c311c5a92ba60f0742f7b9f9d8720bded8f4a2",
                b"provider".as_slice(),
            ),
        ];
        for (bytes, digest, required_contract) in catalogs {
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), digest);
            assert!(
                bytes
                    .windows(required_contract.len())
                    .any(|window| window == required_contract)
            );
        }
    }
}
