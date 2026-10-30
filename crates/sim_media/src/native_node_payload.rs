use sha2::{Digest, Sha256};
use sim_tensor::{DType, DeviceId, StorageId, Tensor, TensorError};
use sim_types::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};
use thiserror::Error;

const MAX_FRAMES: usize = 65_536;
const MAX_ITEMS: usize = 65_536;
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_MEDIA_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_SPLAT_SH_COEFFICIENTS: u64 = 16;
const FACE_LANDMARK_COUNT: usize = 478;
const FACE_BLENDSHAPE_COUNT: usize = 52;

pub const MEDIAPIPE_FACE_BLENDSHAPE_NAMES: [&str; FACE_BLENDSHAPE_COUNT] = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePoint2 {
    x: f64,
    y: f64,
}

impl NativePoint2 {
    pub fn checked(x: f64, y: f64) -> Result<Self, NativeMediaPayloadError> {
        require_finite("point x", x)?;
        require_finite("point y", y)?;
        Ok(Self { x, y })
    }

    pub const fn x(self) -> f64 {
        self.x
    }

    pub const fn y(self) -> f64 {
        self.y
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePoint3 {
    x: f64,
    y: f64,
    z: f64,
}

impl NativePoint3 {
    pub fn checked(x: f64, y: f64, z: f64) -> Result<Self, NativeMediaPayloadError> {
        require_finite("point x", x)?;
        require_finite("point y", y)?;
        require_finite("point z", z)?;
        Ok(Self { x, y, z })
    }

    pub const fn x(self) -> f64 {
        self.x
    }

    pub const fn y(self) -> f64 {
        self.y
    }

    pub const fn z(self) -> f64 {
        self.z
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeBoundingBox {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    label: Option<Box<str>>,
    score: Option<f64>,
}

impl NativeBoundingBox {
    pub fn checked(
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        label: Option<String>,
        score: Option<f64>,
    ) -> Result<Self, NativeMediaPayloadError> {
        require_finite("bounding box x", x)?;
        require_finite("bounding box y", y)?;
        require_non_negative("bounding box width", width)?;
        require_non_negative("bounding box height", height)?;
        let label = label
            .map(|label| checked_text("bounding box label", label))
            .transpose()?
            .map(String::into_boxed_str);
        if let Some(score) = score {
            require_probability("bounding box score", score)?;
        }
        Ok(Self {
            x,
            y,
            width,
            height,
            label,
            score,
        })
    }

    pub const fn x(&self) -> f64 {
        self.x
    }

    pub const fn y(&self) -> f64 {
        self.y
    }

    pub const fn width(&self) -> f64 {
        self.width
    }

    pub const fn height(&self) -> f64 {
        self.height
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub const fn score(&self) -> Option<f64> {
        self.score
    }

    fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        Self::checked(
            self.x,
            self.y,
            self.width,
            self.height,
            self.label.as_deref().map(str::to_owned),
            self.score,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NativeBoundingBoxPayload {
    frames: Box<[Box<[NativeBoundingBox]>]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeBoundingBoxPayload {
    pub const SOURCE_TYPE_ID: &'static str = "BOUNDING_BOX";

    pub fn checked(frames: Vec<Vec<NativeBoundingBox>>) -> Result<Self, NativeMediaPayloadError> {
        check_count("bounding box frames", frames.len(), MAX_FRAMES)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                check_count("bounding boxes per frame", frame.len(), MAX_ITEMS)?;
                for bounding_box in &frame {
                    bounding_box.validate()?;
                }
                Ok(frame.into_boxed_slice())
            })
            .collect::<Result<Vec<_>, NativeMediaPayloadError>>()?
            .into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_bounding_boxes(&frames)?;
        Ok(Self {
            frames,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub fn frames(&self) -> &[Box<[NativeBoundingBox]>] {
        &self.frames
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        let (digest, resident_bytes) = project_bounding_boxes(&self.frames)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeFaceBlendshape {
    name: Box<str>,
    score: f64,
}

impl NativeFaceBlendshape {
    pub fn checked(name: String, score: f64) -> Result<Self, NativeMediaPayloadError> {
        let name = checked_text("face blendshape name", name)?.into_boxed_str();
        require_probability("face blendshape score", score)?;
        Ok(Self { name, score })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeFaceLandmark {
    bbox_xyxy: [f64; 4],
    blendshapes: Box<[NativeFaceBlendshape]>,
    landmarks_xy: Box<[NativePoint2]>,
    landmarks_3d: Box<[NativePoint3]>,
    presence: f64,
    score: f64,
    transformation_matrix: [f64; 16],
}

impl NativeFaceLandmark {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        bbox_xyxy: [f64; 4],
        blendshapes: Vec<NativeFaceBlendshape>,
        landmarks_xy: Vec<NativePoint2>,
        landmarks_3d: Vec<NativePoint3>,
        presence: f64,
        score: f64,
        transformation_matrix: [f64; 16],
    ) -> Result<Self, NativeMediaPayloadError> {
        for value in bbox_xyxy {
            require_finite("face bounding box", value)?;
        }
        if bbox_xyxy[2] < bbox_xyxy[0] || bbox_xyxy[3] < bbox_xyxy[1] {
            return Err(NativeMediaPayloadError::InvalidExtent("face bounding box"));
        }
        check_exact_count("face blendshapes", blendshapes.len(), FACE_BLENDSHAPE_COUNT)?;
        for (index, blendshape) in blendshapes.iter().enumerate() {
            let expected = MEDIAPIPE_FACE_BLENDSHAPE_NAMES[index];
            if blendshape.name() != expected {
                return Err(NativeMediaPayloadError::UnexpectedBlendshape {
                    index,
                    expected,
                    actual: blendshape.name().to_owned(),
                });
            }
            require_probability("face blendshape score", blendshape.score())?;
        }
        check_exact_count("2D face landmarks", landmarks_xy.len(), FACE_LANDMARK_COUNT)?;
        check_exact_count("3D face landmarks", landmarks_3d.len(), FACE_LANDMARK_COUNT)?;
        require_finite("face presence", presence)?;
        require_probability("face score", score)?;
        for value in transformation_matrix {
            require_finite("face transformation matrix", value)?;
        }
        Ok(Self {
            bbox_xyxy,
            blendshapes: blendshapes.into_boxed_slice(),
            landmarks_xy: landmarks_xy.into_boxed_slice(),
            landmarks_3d: landmarks_3d.into_boxed_slice(),
            presence,
            score,
            transformation_matrix,
        })
    }

    pub const fn bbox_xyxy(&self) -> &[f64; 4] {
        &self.bbox_xyxy
    }

    pub fn blendshapes(&self) -> &[NativeFaceBlendshape] {
        &self.blendshapes
    }

    pub fn landmarks_xy(&self) -> &[NativePoint2] {
        &self.landmarks_xy
    }

    pub fn landmarks_3d(&self) -> &[NativePoint3] {
        &self.landmarks_3d
    }

    pub const fn presence(&self) -> f64 {
        self.presence
    }

    pub const fn score(&self) -> f64 {
        self.score
    }

    pub const fn transformation_matrix(&self) -> &[f64; 16] {
        &self.transformation_matrix
    }

    fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        Self::checked(
            self.bbox_xyxy,
            self.blendshapes.to_vec(),
            self.landmarks_xy.to_vec(),
            self.landmarks_3d.to_vec(),
            self.presence,
            self.score,
            self.transformation_matrix,
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeFaceConnectionSet {
    name: Box<str>,
    edges: Box<[[u16; 2]]>,
}

impl NativeFaceConnectionSet {
    pub fn checked(
        name: String,
        mut edges: Vec<[u16; 2]>,
    ) -> Result<Self, NativeMediaPayloadError> {
        let name = checked_text("face connection set name", name)?.into_boxed_str();
        check_count("face connection edges", edges.len(), MAX_ITEMS)?;
        for edge in &edges {
            if usize::from(edge[0]) >= FACE_LANDMARK_COUNT
                || usize::from(edge[1]) >= FACE_LANDMARK_COUNT
                || edge[0] == edge[1]
            {
                return Err(NativeMediaPayloadError::InvalidFaceConnection(*edge));
            }
        }
        edges.sort_unstable();
        if edges.windows(2).any(|window| window[0] == window[1]) {
            return Err(NativeMediaPayloadError::DuplicateFaceConnection);
        }
        Ok(Self {
            name,
            edges: edges.into_boxed_slice(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn edges(&self) -> &[[u16; 2]] {
        &self.edges
    }
}

#[derive(Clone, Debug)]
pub struct NativeFaceLandmarksPayload {
    image_height: u32,
    image_width: u32,
    frames: Box<[Box<[NativeFaceLandmark]>]>,
    connection_sets: Box<[NativeFaceConnectionSet]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeFaceLandmarksPayload {
    pub const SOURCE_TYPE_ID: &'static str = "FACE_LANDMARKS";

    pub fn checked(
        image_height: u32,
        image_width: u32,
        frames: Vec<Vec<NativeFaceLandmark>>,
        mut connection_sets: Vec<NativeFaceConnectionSet>,
    ) -> Result<Self, NativeMediaPayloadError> {
        require_image_size(image_height, image_width)?;
        check_count("face landmark frames", frames.len(), MAX_FRAMES)?;
        let frames = frames
            .into_iter()
            .map(|frame| {
                check_count("faces per frame", frame.len(), MAX_ITEMS)?;
                for face in &frame {
                    face.validate()?;
                }
                Ok(frame.into_boxed_slice())
            })
            .collect::<Result<Vec<_>, NativeMediaPayloadError>>()?
            .into_boxed_slice();
        check_count("face connection sets", connection_sets.len(), MAX_ITEMS)?;
        connection_sets.sort_by(|left, right| left.name.cmp(&right.name));
        if connection_sets
            .windows(2)
            .any(|window| window[0].name == window[1].name)
        {
            return Err(NativeMediaPayloadError::DuplicateConnectionSet);
        }
        let connection_sets = connection_sets.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) =
            project_face_landmarks(image_height, image_width, &frames, &connection_sets)?;
        Ok(Self {
            image_height,
            image_width,
            frames,
            connection_sets,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn image_height(&self) -> u32 {
        self.image_height
    }

    pub const fn image_width(&self) -> u32 {
        self.image_width
    }

    pub fn frames(&self) -> &[Box<[NativeFaceLandmark]>] {
        &self.frames
    }

    pub fn connection_sets(&self) -> &[NativeFaceConnectionSet] {
        &self.connection_sets
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        require_image_size(self.image_height, self.image_width)?;
        let (digest, resident_bytes) = project_face_landmarks(
            self.image_height,
            self.image_width,
            &self.frames,
            &self.connection_sets,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePoseKeypoint {
    x: f64,
    y: f64,
    score: f64,
}

impl NativePoseKeypoint {
    pub fn checked(x: f64, y: f64, score: f64) -> Result<Self, NativeMediaPayloadError> {
        require_finite("pose keypoint x", x)?;
        require_finite("pose keypoint y", y)?;
        require_finite("pose keypoint score", score)?;
        Ok(Self { x, y, score })
    }

    pub const fn x(self) -> f64 {
        self.x
    }

    pub const fn y(self) -> f64 {
        self.y
    }

    pub const fn score(self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativePosePerson {
    pose: Box<[NativePoseKeypoint]>,
    foot: Box<[NativePoseKeypoint]>,
    face: Box<[NativePoseKeypoint]>,
    hand_right: Box<[NativePoseKeypoint]>,
    hand_left: Box<[NativePoseKeypoint]>,
}

impl NativePosePerson {
    pub fn checked(
        pose: Vec<NativePoseKeypoint>,
        foot: Vec<NativePoseKeypoint>,
        face: Vec<NativePoseKeypoint>,
        hand_right: Vec<NativePoseKeypoint>,
        hand_left: Vec<NativePoseKeypoint>,
    ) -> Result<Self, NativeMediaPayloadError> {
        check_exact_count("body pose keypoints", pose.len(), 18)?;
        check_exact_count("foot keypoints", foot.len(), 6)?;
        check_exact_count("face pose keypoints", face.len(), 70)?;
        check_exact_count("right hand keypoints", hand_right.len(), 21)?;
        check_exact_count("left hand keypoints", hand_left.len(), 21)?;
        Ok(Self {
            pose: pose.into_boxed_slice(),
            foot: foot.into_boxed_slice(),
            face: face.into_boxed_slice(),
            hand_right: hand_right.into_boxed_slice(),
            hand_left: hand_left.into_boxed_slice(),
        })
    }

    pub fn pose(&self) -> &[NativePoseKeypoint] {
        &self.pose
    }

    pub fn foot(&self) -> &[NativePoseKeypoint] {
        &self.foot
    }

    pub fn face(&self) -> &[NativePoseKeypoint] {
        &self.face
    }

    pub fn hand_right(&self) -> &[NativePoseKeypoint] {
        &self.hand_right
    }

    pub fn hand_left(&self) -> &[NativePoseKeypoint] {
        &self.hand_left
    }

    fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        check_exact_count("body pose keypoints", self.pose.len(), 18)?;
        check_exact_count("foot keypoints", self.foot.len(), 6)?;
        check_exact_count("face pose keypoints", self.face.len(), 70)?;
        check_exact_count("right hand keypoints", self.hand_right.len(), 21)?;
        check_exact_count("left hand keypoints", self.hand_left.len(), 21)?;
        for point in self
            .pose
            .iter()
            .chain(self.foot.iter())
            .chain(self.face.iter())
            .chain(self.hand_right.iter())
            .chain(self.hand_left.iter())
        {
            NativePoseKeypoint::checked(point.x, point.y, point.score)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativePoseFrame {
    canvas_width: u32,
    canvas_height: u32,
    people: Box<[NativePosePerson]>,
}

impl NativePoseFrame {
    pub fn checked(
        canvas_width: u32,
        canvas_height: u32,
        people: Vec<NativePosePerson>,
    ) -> Result<Self, NativeMediaPayloadError> {
        require_image_size(canvas_height, canvas_width)?;
        check_count("people per pose frame", people.len(), MAX_ITEMS)?;
        for person in &people {
            person.validate()?;
        }
        Ok(Self {
            canvas_width,
            canvas_height,
            people: people.into_boxed_slice(),
        })
    }

    pub const fn canvas_width(&self) -> u32 {
        self.canvas_width
    }

    pub const fn canvas_height(&self) -> u32 {
        self.canvas_height
    }

    pub fn people(&self) -> &[NativePosePerson] {
        &self.people
    }
}

#[derive(Clone, Debug)]
pub struct NativePoseKeypointPayload {
    frames: Box<[NativePoseFrame]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativePoseKeypointPayload {
    pub const SOURCE_TYPE_ID: &'static str = "POSE_KEYPOINT";

    pub fn checked(frames: Vec<NativePoseFrame>) -> Result<Self, NativeMediaPayloadError> {
        check_count("pose frames", frames.len(), MAX_FRAMES)?;
        for frame in &frames {
            require_image_size(frame.canvas_height, frame.canvas_width)?;
            for person in frame.people.iter() {
                person.validate()?;
            }
        }
        let frames = frames.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_pose_keypoints(&frames)?;
        Ok(Self {
            frames,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub fn frames(&self) -> &[NativePoseFrame] {
        &self.frames
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        let (digest, resident_bytes) = project_pose_keypoints(&self.frames)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NativeTracksPayload {
    track_path: Tensor,
    track_visibility: Tensor,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMediaTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl NativeMediaTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMediaResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<NativeMediaTensorResidentAllocation>,
}

impl NativeMediaResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[NativeMediaTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeMediaPayloadError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeArtifactKind {
    AudioRecord,
    Svg,
    Webcam,
}

impl NativeArtifactKind {
    pub const fn source_type_id(self) -> &'static str {
        match self {
            Self::AudioRecord => "AUDIO_RECORD",
            Self::Svg => "SVG",
            Self::Webcam => "WEBCAM",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeArtifactPayload {
    kind: NativeArtifactKind,
    media_type: Box<str>,
    bytes: Box<[u8]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeArtifactPayload {
    pub fn checked(
        kind: NativeArtifactKind,
        media_type: String,
        bytes: Vec<u8>,
    ) -> Result<Self, NativeMediaPayloadError> {
        let media_type = checked_media_type(media_type)?.into_boxed_str();
        check_byte_count("artifact bytes", bytes.len())?;
        let bytes = bytes.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_byte_asset::<Self>(
            b"zed.sim.media.artifact.v1",
            &[kind_tag(kind)],
            &media_type,
            &bytes,
        )?;
        Ok(Self {
            kind,
            media_type,
            bytes,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn kind(&self) -> NativeArtifactKind {
        self.kind
    }

    pub const fn source_type_id(&self) -> &'static str {
        self.kind.source_type_id()
    }

    pub fn media_type(&self) -> &str {
        &self.media_type
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        checked_media_type(self.media_type.to_string())?;
        check_byte_count("artifact bytes", self.bytes.len())?;
        let (digest, resident_bytes) = project_byte_asset::<Self>(
            b"zed.sim.media.artifact.v1",
            &[kind_tag(self.kind)],
            &self.media_type,
            &self.bytes,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFile3DFormat {
    Fbx,
    Gltf,
    Glb,
    Ksplat,
    Obj,
    Ply,
    Splat,
    Spz,
    Stl,
    Usdz,
}

impl NativeFile3DFormat {
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Fbx => "fbx",
            Self::Gltf => "gltf",
            Self::Glb => "glb",
            Self::Ksplat => "ksplat",
            Self::Obj => "obj",
            Self::Ply => "ply",
            Self::Splat => "splat",
            Self::Spz => "spz",
            Self::Stl => "stl",
            Self::Usdz => "usdz",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeFile3DRole {
    Any,
    PointCloudAny,
    SplatAny,
    Fbx,
    Gltf,
    Glb,
    Ksplat,
    Obj,
    Ply,
    Splat,
    Spz,
    Stl,
    Usdz,
}

impl NativeFile3DRole {
    pub const fn source_type_id(self) -> &'static str {
        match self {
            Self::Any => "FILE_3D",
            Self::PointCloudAny => "FILE_3D_POINT_CLOUD_ANY",
            Self::SplatAny => "FILE_3D_SPLAT_ANY",
            Self::Fbx => "FILE_3D_FBX",
            Self::Gltf => "FILE_3D_GLTF",
            Self::Glb => "FILE_3D_GLB",
            Self::Ksplat => "FILE_3D_KSPLAT",
            Self::Obj => "FILE_3D_OBJ",
            Self::Ply => "FILE_3D_PLY",
            Self::Splat => "FILE_3D_SPLAT",
            Self::Spz => "FILE_3D_SPZ",
            Self::Stl => "FILE_3D_STL",
            Self::Usdz => "FILE_3D_USDZ",
        }
    }

    const fn accepts(self, format: NativeFile3DFormat) -> bool {
        match self {
            Self::Any => true,
            Self::PointCloudAny => matches!(format, NativeFile3DFormat::Ply),
            Self::SplatAny => matches!(
                format,
                NativeFile3DFormat::Ksplat
                    | NativeFile3DFormat::Ply
                    | NativeFile3DFormat::Splat
                    | NativeFile3DFormat::Spz
            ),
            Self::Fbx => matches!(format, NativeFile3DFormat::Fbx),
            Self::Gltf => matches!(format, NativeFile3DFormat::Gltf),
            Self::Glb => matches!(format, NativeFile3DFormat::Glb),
            Self::Ksplat => matches!(format, NativeFile3DFormat::Ksplat),
            Self::Obj => matches!(format, NativeFile3DFormat::Obj),
            Self::Ply => matches!(format, NativeFile3DFormat::Ply),
            Self::Splat => matches!(format, NativeFile3DFormat::Splat),
            Self::Spz => matches!(format, NativeFile3DFormat::Spz),
            Self::Stl => matches!(format, NativeFile3DFormat::Stl),
            Self::Usdz => matches!(format, NativeFile3DFormat::Usdz),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeFile3DPayload {
    role: NativeFile3DRole,
    format: NativeFile3DFormat,
    bytes: Box<[u8]>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeFile3DPayload {
    pub fn checked(
        role: NativeFile3DRole,
        format: NativeFile3DFormat,
        bytes: Vec<u8>,
    ) -> Result<Self, NativeMediaPayloadError> {
        if !role.accepts(format) {
            return Err(NativeMediaPayloadError::InvalidFile3DRole);
        }
        check_byte_count("3D file bytes", bytes.len())?;
        if bytes.is_empty() {
            return Err(NativeMediaPayloadError::EmptyMedia("3D file"));
        }
        validate_file_3d_contents(format, &bytes)?;
        let bytes = bytes.into_boxed_slice();
        let tags = [file_role_tag(role), file_format_tag(format)];
        let (semantic_digest_sha256, resident_bytes) = project_byte_asset::<Self>(
            b"zed.sim.media.file-3d.v1",
            &tags,
            format.extension(),
            &bytes,
        )?;
        Ok(Self {
            role,
            format,
            bytes,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn role(&self) -> NativeFile3DRole {
        self.role
    }

    pub const fn format(&self) -> NativeFile3DFormat {
        self.format
    }

    pub const fn source_type_id(&self) -> &'static str {
        self.role.source_type_id()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        if !self.role.accepts(self.format) {
            return Err(NativeMediaPayloadError::InvalidFile3DRole);
        }
        check_byte_count("3D file bytes", self.bytes.len())?;
        if self.bytes.is_empty() {
            return Err(NativeMediaPayloadError::EmptyMedia("3D file"));
        }
        validate_file_3d_contents(self.format, &self.bytes)?;
        let tags = [file_role_tag(self.role), file_format_tag(self.format)];
        let (digest, resident_bytes) = project_byte_asset::<Self>(
            b"zed.sim.media.file-3d.v1",
            &tags,
            self.format.extension(),
            &self.bytes,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NativeAudioPayload {
    waveform: Tensor,
    sample_rate: u32,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeAudioPayload {
    pub const SOURCE_TYPE_ID: &'static str = "AUDIO";

    pub fn checked(waveform: Tensor, sample_rate: u32) -> Result<Self, NativeMediaPayloadError> {
        validate_audio(&waveform, sample_rate)?;
        let (semantic_digest_sha256, resident_bytes) = project_audio(&waveform, sample_rate)?;
        Ok(Self {
            waveform,
            sample_rate,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub fn checked_cancellable(
        waveform: Tensor,
        sample_rate: u32,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeCancellableMediaPayloadError> {
        validate_audio_cancellable(&waveform, sample_rate, cancellation)?;
        let (semantic_digest_sha256, resident_bytes) =
            project_audio_cancellable(&waveform, sample_rate, cancellation)?;
        Ok(Self {
            waveform,
            sample_rate,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn waveform(&self) -> &Tensor {
        &self.waveform
    }

    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        exact_tensor_parts::<Self, 1>([&self.waveform], self.resident_bytes)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_audio(&self.waveform, self.sample_rate)?;
        let (digest, resident_bytes) = project_audio(&self.waveform, self.sample_rate)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }

    pub fn validate_cancellable(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeCancellableMediaPayloadError> {
        validate_audio_cancellable(&self.waveform, self.sample_rate, cancellation)?;
        let (digest, resident_bytes) =
            project_audio_cancellable(&self.waveform, self.sample_rate, cancellation)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        check_media_cancellation(cancellation)
    }
}

#[derive(Clone, Debug)]
pub enum NativeVideoPayload {
    Components(NativeVideoComponentsPayload),
    Encoded(NativeEncodedVideoPayload),
}

#[derive(Clone, Debug)]
pub struct NativeVideoComponentsPayload {
    frames: Tensor,
    frame_rate_numerator: u64,
    frame_rate_denominator: u64,
    bit_depth: NativeVideoBitDepth,
    audio: Option<NativeAudioPayload>,
    alpha: Option<Tensor>,
    metadata: BTreeMap<String, String>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct NativeEncodedVideoPayload {
    bytes: Tensor,
    content_sha256: [u8; 32],
    source_video_sha256: [u8; 32],
    dimensions: (u64, u64),
    frame_rate: (u64, u64),
    frame_count: u64,
    bit_depth: NativeVideoBitDepth,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoRepresentation {
    Components,
    Encoded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVideoBitDepth {
    Eight,
    Ten,
}

impl NativeVideoBitDepth {
    pub const fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
        }
    }
}

impl TryFrom<u8> for NativeVideoBitDepth {
    type Error = NativeMediaPayloadError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(Self::Eight),
            10 => Ok(Self::Ten),
            _ => Err(NativeMediaPayloadError::InvalidVideo),
        }
    }
}

impl NativeVideoPayload {
    pub const SOURCE_TYPE_ID: &'static str = "VIDEO";

    pub fn checked(
        frames: Tensor,
        frame_rate_numerator: u64,
        frame_rate_denominator: u64,
        bit_depth: NativeVideoBitDepth,
        audio: Option<NativeAudioPayload>,
        alpha: Option<Tensor>,
        metadata: BTreeMap<String, String>,
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_video(
            &frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio.as_ref(),
            alpha.as_ref(),
            &metadata,
        )?;
        let (semantic_digest_sha256, resident_bytes) = project_video(
            &frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio.as_ref(),
            alpha.as_ref(),
            &metadata,
        )?;
        Ok(Self::Components(NativeVideoComponentsPayload {
            frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio,
            alpha,
            metadata,
            semantic_digest_sha256,
            resident_bytes,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn checked_cancellable(
        frames: Tensor,
        frame_rate_numerator: u64,
        frame_rate_denominator: u64,
        bit_depth: NativeVideoBitDepth,
        audio: Option<NativeAudioPayload>,
        alpha: Option<Tensor>,
        metadata: BTreeMap<String, String>,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeCancellableMediaPayloadError> {
        validate_video_cancellable(
            &frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio.as_ref(),
            alpha.as_ref(),
            &metadata,
            cancellation,
        )?;
        let (semantic_digest_sha256, resident_bytes) = project_video_cancellable(
            &frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio.as_ref(),
            alpha.as_ref(),
            &metadata,
            cancellation,
        )?;
        Ok(Self::Components(NativeVideoComponentsPayload {
            frames,
            frame_rate_numerator,
            frame_rate_denominator,
            bit_depth,
            audio,
            alpha,
            metadata,
            semantic_digest_sha256,
            resident_bytes,
        }))
    }

    pub fn checked_h264_mp4_from_component(
        source: &NativeVideoPayload,
        bytes: Tensor,
        content_sha256: [u8; 32],
        dimensions: (u64, u64),
        frame_rate: (u64, u64),
        frame_count: u64,
    ) -> Result<Self, NativeMediaPayloadError> {
        source.validate()?;
        let components = source
            .components()
            .ok_or(NativeMediaPayloadError::InvalidVideo)?;
        let [
            source_frame_count,
            source_height,
            source_width,
            source_channels,
        ] = components.frames().descriptor().shape()
        else {
            return Err(NativeMediaPayloadError::InvalidVideo);
        };
        let bit_depth = components.bit_depth();
        if components.frames().descriptor().dtype() != DType::F32
            || components.frames().descriptor().device() != DeviceId::CPU
            || !components.frames().descriptor().is_contiguous()?
            || !matches!(*source_channels, 3 | 4)
            || components.audio().is_some()
            || (*source_width, *source_height) != dimensions
            || *source_frame_count != frame_count
        {
            return Err(NativeMediaPayloadError::InvalidVideo);
        }
        let source_video_sha256 = *source.semantic_digest_sha256();
        validate_encoded_h264_mp4(&bytes, content_sha256, dimensions, frame_rate, frame_count)?;
        let (semantic_digest_sha256, resident_bytes) = project_encoded_h264_mp4(
            &bytes,
            content_sha256,
            source_video_sha256,
            dimensions,
            frame_rate,
            frame_count,
            bit_depth,
        )?;
        Ok(Self::Encoded(NativeEncodedVideoPayload {
            bytes,
            content_sha256,
            source_video_sha256,
            dimensions,
            frame_rate,
            frame_count,
            bit_depth,
            semantic_digest_sha256,
            resident_bytes,
        }))
    }

    pub const fn representation(&self) -> NativeVideoRepresentation {
        match self {
            Self::Components(_) => NativeVideoRepresentation::Components,
            Self::Encoded(_) => NativeVideoRepresentation::Encoded,
        }
    }

    pub const fn components(&self) -> Option<&NativeVideoComponentsPayload> {
        match self {
            Self::Components(components) => Some(components),
            Self::Encoded(_) => None,
        }
    }

    pub const fn encoded(&self) -> Option<&NativeEncodedVideoPayload> {
        match self {
            Self::Components(_) => None,
            Self::Encoded(encoded) => Some(encoded),
        }
    }

    pub const fn frame_rate(&self) -> (u64, u64) {
        match self {
            Self::Components(components) => components.frame_rate(),
            Self::Encoded(encoded) => encoded.frame_rate(),
        }
    }

    pub const fn bit_depth(&self) -> NativeVideoBitDepth {
        match self {
            Self::Components(components) => components.bit_depth(),
            Self::Encoded(encoded) => encoded.bit_depth(),
        }
    }

    pub fn duration_seconds(&self) -> f64 {
        match self {
            Self::Components(components) => components.duration_seconds(),
            Self::Encoded(encoded) => encoded.duration_seconds(),
        }
    }

    pub fn dimensions(&self) -> (u64, u64) {
        match self {
            Self::Components(components) => components.dimensions(),
            Self::Encoded(encoded) => encoded.dimensions(),
        }
    }

    pub fn frame_count(&self) -> u64 {
        match self {
            Self::Components(components) => components.frame_count(),
            Self::Encoded(encoded) => encoded.frame_count(),
        }
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        match self {
            Self::Components(components) => components.semantic_digest_sha256(),
            Self::Encoded(encoded) => encoded.semantic_digest_sha256(),
        }
    }

    pub const fn resident_bytes(&self) -> u64 {
        match self {
            Self::Components(components) => components.resident_bytes(),
            Self::Encoded(encoded) => encoded.resident_bytes(),
        }
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        match self {
            Self::Components(components) => components.resident_parts(),
            Self::Encoded(encoded) => encoded.resident_parts(),
        }
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        match self {
            Self::Components(components) => components.validate(),
            Self::Encoded(encoded) => encoded.validate(),
        }
    }

    pub fn validate_components_cancellable(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeCancellableMediaPayloadError> {
        let Self::Components(components) = self else {
            return Err(NativeMediaPayloadError::InvalidVideo.into());
        };
        validate_video_cancellable(
            &components.frames,
            components.frame_rate_numerator,
            components.frame_rate_denominator,
            components.bit_depth,
            components.audio.as_ref(),
            components.alpha.as_ref(),
            &components.metadata,
            cancellation,
        )?;
        let (digest, resident_bytes) = project_video_cancellable(
            &components.frames,
            components.frame_rate_numerator,
            components.frame_rate_denominator,
            components.bit_depth,
            components.audio.as_ref(),
            components.alpha.as_ref(),
            &components.metadata,
            cancellation,
        )?;
        require_projection(
            components.semantic_digest_sha256,
            digest,
            components.resident_bytes,
            resident_bytes,
        )?;
        components.resident_parts()?;
        check_media_cancellation(cancellation)
    }
}

impl NativeVideoComponentsPayload {
    pub const fn frames(&self) -> &Tensor {
        &self.frames
    }

    pub const fn frame_rate(&self) -> (u64, u64) {
        (self.frame_rate_numerator, self.frame_rate_denominator)
    }

    pub const fn bit_depth(&self) -> NativeVideoBitDepth {
        self.bit_depth
    }

    pub fn duration_seconds(&self) -> f64 {
        self.frame_count() as f64 * self.frame_rate_denominator as f64
            / self.frame_rate_numerator as f64
    }

    pub fn dimensions(&self) -> (u64, u64) {
        let shape = self.frames.descriptor().shape();
        (shape[2], shape[1])
    }

    pub fn frame_count(&self) -> u64 {
        self.frames.descriptor().shape()[0]
    }

    pub const fn audio(&self) -> Option<&NativeAudioPayload> {
        self.audio.as_ref()
    }

    pub const fn alpha(&self) -> Option<&Tensor> {
        self.alpha.as_ref()
    }

    pub const fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let metadata_bytes = self
            .metadata
            .iter()
            .try_fold(0_usize, |total, (key, value)| {
                total
                    .checked_add(key.capacity())
                    .and_then(|total| total.checked_add(value.capacity()))
                    .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)
            })?;
        let owned_bytes = mem::size_of::<NativeVideoPayload>()
            .checked_add(metadata_bytes)
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        let mut tensors = vec![&self.frames];
        if let Some(audio) = &self.audio {
            tensors.push(audio.waveform());
        }
        if let Some(alpha) = &self.alpha {
            tensors.push(alpha);
        }
        exact_tensor_parts_with_owned(owned_bytes, tensors, self.resident_bytes)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_video(
            &self.frames,
            self.frame_rate_numerator,
            self.frame_rate_denominator,
            self.bit_depth,
            self.audio.as_ref(),
            self.alpha.as_ref(),
            &self.metadata,
        )?;
        let (digest, resident_bytes) = project_video(
            &self.frames,
            self.frame_rate_numerator,
            self.frame_rate_denominator,
            self.bit_depth,
            self.audio.as_ref(),
            self.alpha.as_ref(),
            &self.metadata,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

impl NativeEncodedVideoPayload {
    pub const fn bytes(&self) -> &Tensor {
        &self.bytes
    }

    pub const fn content_sha256(&self) -> &[u8; 32] {
        &self.content_sha256
    }

    pub const fn source_video_sha256(&self) -> &[u8; 32] {
        &self.source_video_sha256
    }

    pub const fn dimensions(&self) -> (u64, u64) {
        self.dimensions
    }

    pub const fn frame_rate(&self) -> (u64, u64) {
        self.frame_rate
    }

    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn duration_seconds(&self) -> f64 {
        self.frame_count as f64 * self.frame_rate.1 as f64 / self.frame_rate.0 as f64
    }

    pub const fn container(&self) -> crate::NativeVideoContainer {
        crate::NativeVideoContainer::Mp4
    }

    pub const fn codec(&self) -> crate::NativeVideoCodec {
        crate::NativeVideoCodec::H264
    }

    pub const fn pixel_format(&self) -> crate::NativeVideoPixelFormat {
        match self.bit_depth {
            NativeVideoBitDepth::Eight => crate::NativeVideoPixelFormat::Yuv420p,
            NativeVideoBitDepth::Ten => crate::NativeVideoPixelFormat::Yuv420p10le,
        }
    }

    pub const fn bit_depth(&self) -> NativeVideoBitDepth {
        self.bit_depth
    }

    pub const fn has_audio(&self) -> bool {
        false
    }

    pub const fn has_alpha(&self) -> bool {
        false
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        exact_tensor_parts_with_owned(
            mem::size_of::<NativeVideoPayload>(),
            [&self.bytes],
            self.resident_bytes,
        )
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_encoded_h264_mp4(
            &self.bytes,
            self.content_sha256,
            self.dimensions,
            self.frame_rate,
            self.frame_count,
        )?;
        let (digest, resident_bytes) = project_encoded_h264_mp4(
            &self.bytes,
            self.content_sha256,
            self.source_video_sha256,
            self.dimensions,
            self.frame_rate,
            self.frame_count,
            self.bit_depth,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCameraRole {
    CameraControl,
    Load3D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NativeCameraProjection {
    Perspective {
        fov_degrees: f32,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

impl NativeCameraRole {
    pub const fn source_type_id(self) -> &'static str {
        match self {
            Self::CameraControl => "CAMERA_CONTROL",
            Self::Load3D => "LOAD3D_CAMERA",
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeCameraPayload {
    role: NativeCameraRole,
    position: [f32; 3],
    target: [f32; 3],
    zoom: f32,
    orientation_wxyz: Option<[f32; 4]>,
    projection: NativeCameraProjection,
    width: u32,
    height: u32,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeCameraPayload {
    pub fn checked(
        role: NativeCameraRole,
        position: [f32; 3],
        target: [f32; 3],
        zoom: f32,
        orientation_wxyz: Option<[f32; 4]>,
        projection: NativeCameraProjection,
        width: u32,
        height: u32,
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_camera(
            &position,
            &target,
            zoom,
            orientation_wxyz.as_ref(),
            projection,
            width,
            height,
        )?;
        let (semantic_digest_sha256, resident_bytes) = project_camera::<Self>(
            role,
            &position,
            &target,
            zoom,
            orientation_wxyz.as_ref(),
            projection,
            width,
            height,
        )?;
        Ok(Self {
            role,
            position,
            target,
            zoom,
            orientation_wxyz,
            projection,
            width,
            height,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn role(&self) -> NativeCameraRole {
        self.role
    }

    pub const fn source_type_id(&self) -> &'static str {
        self.role.source_type_id()
    }

    pub const fn position(&self) -> &[f32; 3] {
        &self.position
    }

    pub const fn target(&self) -> &[f32; 3] {
        &self.target
    }

    pub const fn zoom(&self) -> f32 {
        self.zoom
    }

    pub const fn orientation_wxyz(&self) -> Option<&[f32; 4]> {
        self.orientation_wxyz.as_ref()
    }

    pub const fn projection(&self) -> NativeCameraProjection {
        self.projection
    }

    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_camera(
            &self.position,
            &self.target,
            self.zoom,
            self.orientation_wxyz.as_ref(),
            self.projection,
            self.width,
            self.height,
        )?;
        let (digest, resident_bytes) = project_camera::<Self>(
            self.role,
            &self.position,
            &self.target,
            self.zoom,
            self.orientation_wxyz.as_ref(),
            self.projection,
            self.width,
            self.height,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NativeSplatPayload {
    positions: Tensor,
    scales: Tensor,
    rotations: Tensor,
    opacity: Tensor,
    spherical_harmonics: Tensor,
    counts: Option<Box<[u64]>>,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeSplatPayload {
    pub const SOURCE_TYPE_ID: &'static str = "SPLAT";

    pub fn checked(
        positions: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacity: Tensor,
        spherical_harmonics: Tensor,
        counts: Option<Vec<u64>>,
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_splat(
            &positions,
            &scales,
            &rotations,
            &opacity,
            &spherical_harmonics,
            counts.as_deref(),
        )?;
        let padded_count = positions.descriptor().shape()[1];
        let counts = counts
            .filter(|counts| counts.iter().any(|count| *count != padded_count))
            .map(Vec::into_boxed_slice);
        let (semantic_digest_sha256, resident_bytes) = project_splat::<Self>(
            &positions,
            &scales,
            &rotations,
            &opacity,
            &spherical_harmonics,
            counts.as_deref(),
        )?;
        Ok(Self {
            positions,
            scales,
            rotations,
            opacity,
            spherical_harmonics,
            counts,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn positions(&self) -> &Tensor {
        &self.positions
    }

    pub const fn scales(&self) -> &Tensor {
        &self.scales
    }

    pub const fn rotations(&self) -> &Tensor {
        &self.rotations
    }

    pub const fn opacity(&self) -> &Tensor {
        &self.opacity
    }

    pub const fn spherical_harmonics(&self) -> &Tensor {
        &self.spherical_harmonics
    }

    pub fn counts(&self) -> Option<&[u64]> {
        self.counts.as_deref()
    }

    pub fn batch_count(&self) -> u64 {
        self.positions.descriptor().shape()[0]
    }

    pub fn splat_count(&self) -> u64 {
        self.counts.as_deref().map_or_else(
            || self.positions.descriptor().shape()[1],
            |counts| counts.iter().copied().sum(),
        )
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let owned_bytes = mem::size_of::<Self>()
            .checked_add(self.counts.as_deref().map_or(Ok(0), |counts| {
                counts
                    .len()
                    .checked_mul(mem::size_of::<u64>())
                    .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)
            })?)
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        exact_tensor_parts_with_owned(
            owned_bytes,
            [
                &self.positions,
                &self.scales,
                &self.rotations,
                &self.opacity,
                &self.spherical_harmonics,
            ],
            self.resident_bytes,
        )
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_splat(
            &self.positions,
            &self.scales,
            &self.rotations,
            &self.opacity,
            &self.spherical_harmonics,
            self.counts.as_deref(),
        )?;
        let (digest, resident_bytes) = project_splat::<Self>(
            &self.positions,
            &self.scales,
            &self.rotations,
            &self.opacity,
            &self.spherical_harmonics,
            self.counts.as_deref(),
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NativeMeshBatch {
    vertices: Tensor,
    faces: Tensor,
    normals: Option<Tensor>,
    uvs: Option<Tensor>,
    colors: Option<Tensor>,
    texture: Option<Tensor>,
}

impl NativeMeshBatch {
    pub fn checked(
        vertices: Tensor,
        faces: Tensor,
        normals: Option<Tensor>,
        uvs: Option<Tensor>,
        colors: Option<Tensor>,
        texture: Option<Tensor>,
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_mesh_batch(
            &vertices,
            &faces,
            normals.as_ref(),
            uvs.as_ref(),
            colors.as_ref(),
            texture.as_ref(),
        )?;
        Ok(Self {
            vertices,
            faces,
            normals,
            uvs,
            colors,
            texture,
        })
    }

    pub const fn vertices(&self) -> &Tensor {
        &self.vertices
    }

    pub const fn faces(&self) -> &Tensor {
        &self.faces
    }

    pub const fn normals(&self) -> Option<&Tensor> {
        self.normals.as_ref()
    }

    pub const fn colors(&self) -> Option<&Tensor> {
        self.colors.as_ref()
    }

    pub const fn uvs(&self) -> Option<&Tensor> {
        self.uvs.as_ref()
    }

    pub const fn texture(&self) -> Option<&Tensor> {
        self.texture.as_ref()
    }

    fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_mesh_batch(
            &self.vertices,
            &self.faces,
            self.normals.as_ref(),
            self.uvs.as_ref(),
            self.colors.as_ref(),
            self.texture.as_ref(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct NativeMeshPayload {
    batches: Box<[NativeMeshBatch]>,
    unlit: bool,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeMeshPayload {
    pub const SOURCE_TYPE_ID: &'static str = "MESH";

    pub fn checked(
        batches: Vec<NativeMeshBatch>,
        unlit: bool,
    ) -> Result<Self, NativeMediaPayloadError> {
        if batches.is_empty() {
            return Err(NativeMediaPayloadError::EmptyMedia("mesh batches"));
        }
        check_count("mesh batches", batches.len(), MAX_ITEMS)?;
        for batch in &batches {
            batch.validate()?;
        }
        let batches = batches.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_mesh::<Self>(&batches, unlit)?;
        Ok(Self {
            batches,
            unlit,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub fn batches(&self) -> &[NativeMeshBatch] {
        &self.batches
    }

    pub const fn unlit(&self) -> bool {
        self.unlit
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let owned_bytes = mem::size_of::<Self>()
            .checked_add(
                mem::size_of::<NativeMeshBatch>()
                    .checked_mul(self.batches.len())
                    .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?,
            )
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        let mut tensors = Vec::new();
        for batch in &self.batches {
            tensors.push(&batch.vertices);
            tensors.push(&batch.faces);
            if let Some(normals) = &batch.normals {
                tensors.push(normals);
            }
            if let Some(uvs) = &batch.uvs {
                tensors.push(uvs);
            }
            if let Some(colors) = &batch.colors {
                tensors.push(colors);
            }
            if let Some(texture) = &batch.texture {
                tensors.push(texture);
            }
        }
        exact_tensor_parts_with_owned(owned_bytes, tensors, self.resident_bytes)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        if self.batches.is_empty() {
            return Err(NativeMediaPayloadError::EmptyMedia("mesh batches"));
        }
        check_count("mesh batches", self.batches.len(), MAX_ITEMS)?;
        for batch in &self.batches {
            batch.validate()?;
        }
        let (digest, resident_bytes) = project_mesh::<Self>(&self.batches, self.unlit)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NativeVoxelPayload {
    density: Tensor,
    colors: Option<Tensor>,
    world_from_grid: [f32; 16],
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeVoxelPayload {
    pub const SOURCE_TYPE_ID: &'static str = "VOXEL";

    pub fn checked(
        density: Tensor,
        colors: Option<Tensor>,
        world_from_grid: [f32; 16],
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_voxel(&density, colors.as_ref(), &world_from_grid)?;
        let (semantic_digest_sha256, resident_bytes) =
            project_voxel::<Self>(&density, colors.as_ref(), &world_from_grid)?;
        Ok(Self {
            density,
            colors,
            world_from_grid,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn density(&self) -> &Tensor {
        &self.density
    }

    pub const fn colors(&self) -> Option<&Tensor> {
        self.colors.as_ref()
    }

    pub const fn world_from_grid(&self) -> &[f32; 16] {
        &self.world_from_grid
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let mut tensors = vec![&self.density];
        if let Some(colors) = &self.colors {
            tensors.push(colors);
        }
        exact_tensor_parts_with_owned(mem::size_of::<Self>(), tensors, self.resident_bytes)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_voxel(&self.density, self.colors.as_ref(), &self.world_from_grid)?;
        let (digest, resident_bytes) =
            project_voxel::<Self>(&self.density, self.colors.as_ref(), &self.world_from_grid)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

impl NativeTracksPayload {
    pub const SOURCE_TYPE_ID: &'static str = "TRACKS";

    pub fn checked(
        track_path: Tensor,
        track_visibility: Tensor,
    ) -> Result<Self, NativeMediaPayloadError> {
        validate_tracks(&track_path, &track_visibility)?;
        let (semantic_digest_sha256, resident_bytes) =
            project_tracks(&track_path, &track_visibility)?;
        Ok(Self {
            track_path,
            track_visibility,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn track_path(&self) -> &Tensor {
        &self.track_path
    }

    pub const fn track_visibility(&self) -> &Tensor {
        &self.track_visibility
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let owned_bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?;
        let parts = media_resident_parts(owned_bytes, [&self.track_path, &self.track_visibility])?;
        if parts.resident_bytes()? != self.resident_bytes {
            return Err(NativeMediaPayloadError::ProjectionChanged);
        }
        Ok(parts)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        validate_tracks(&self.track_path, &self.track_visibility)?;
        let (digest, resident_bytes) = project_tracks(&self.track_path, &self.track_visibility)?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NativeSam3TrackDataPayload {
    packed_masks: Option<Tensor>,
    frame_count: u64,
    scores: Box<[f64]>,
    original_height: u32,
    original_width: u32,
    semantic_digest_sha256: [u8; 32],
    resident_bytes: u64,
}

impl NativeSam3TrackDataPayload {
    pub const SOURCE_TYPE_ID: &'static str = "SAM3_TRACK_DATA";

    pub fn checked(
        packed_masks: Option<Tensor>,
        frame_count: u64,
        scores: Vec<f64>,
        original_height: u32,
        original_width: u32,
    ) -> Result<Self, NativeMediaPayloadError> {
        require_image_size(original_height, original_width)?;
        validate_sam3_track_data(packed_masks.as_ref(), frame_count, &scores)?;
        let scores = scores.into_boxed_slice();
        let (semantic_digest_sha256, resident_bytes) = project_sam3_track_data(
            packed_masks.as_ref(),
            frame_count,
            &scores,
            original_height,
            original_width,
        )?;
        Ok(Self {
            packed_masks,
            frame_count,
            scores,
            original_height,
            original_width,
            semantic_digest_sha256,
            resident_bytes,
        })
    }

    pub const fn packed_masks(&self) -> Option<&Tensor> {
        self.packed_masks.as_ref()
    }

    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    pub const fn original_height(&self) -> u32 {
        self.original_height
    }

    pub const fn original_width(&self) -> u32 {
        self.original_width
    }

    pub const fn semantic_digest_sha256(&self) -> &[u8; 32] {
        &self.semantic_digest_sha256
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_parts(&self) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
        let scores_bytes = mem::size_of::<f64>()
            .checked_mul(self.scores.len())
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        let owned_bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?
            .checked_add(
                u64::try_from(scores_bytes)
                    .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?,
            )
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        let parts = media_resident_parts(owned_bytes, self.packed_masks.iter())?;
        if parts.resident_bytes()? != self.resident_bytes {
            return Err(NativeMediaPayloadError::ProjectionChanged);
        }
        Ok(parts)
    }

    pub fn validate(&self) -> Result<(), NativeMediaPayloadError> {
        require_image_size(self.original_height, self.original_width)?;
        validate_sam3_track_data(self.packed_masks.as_ref(), self.frame_count, &self.scores)?;
        let (digest, resident_bytes) = project_sam3_track_data(
            self.packed_masks.as_ref(),
            self.frame_count,
            &self.scores,
            self.original_height,
            self.original_width,
        )?;
        require_projection(
            self.semantic_digest_sha256,
            digest,
            self.resident_bytes,
            resident_bytes,
        )?;
        self.resident_parts()?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum NativeMediaPayloadError {
    #[error("{field} contains a non-finite value")]
    NonFinite { field: &'static str },
    #[error("{0} must be non-negative")]
    Negative(&'static str),
    #[error("{0} must be in the inclusive range 0..=1")]
    NotProbability(&'static str),
    #[error("{field} contains {actual} items, maximum {maximum}")]
    TooMany {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("{field} contains {actual} items, expected exactly {expected}")]
    WrongCardinality {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("{0} must contain 1..={MAX_TEXT_BYTES} non-control UTF-8 bytes")]
    InvalidText(&'static str),
    #[error("{0} dimensions must both be nonzero")]
    InvalidImageSize(&'static str),
    #[error("{0} has reversed extents")]
    InvalidExtent(&'static str),
    #[error("blendshape {index} is `{actual}`, expected `{expected}`")]
    UnexpectedBlendshape {
        index: usize,
        expected: &'static str,
        actual: String,
    },
    #[error("face connection {0:?} is invalid")]
    InvalidFaceConnection([u16; 2]),
    #[error("face connection is duplicated")]
    DuplicateFaceConnection,
    #[error("face connection-set name is duplicated")]
    DuplicateConnectionSet,
    #[error("TRACKS tensor descriptors do not match the source contract")]
    InvalidTracksShape,
    #[error("SAM3_TRACK_DATA fields do not match the source contract")]
    InvalidSam3TrackData,
    #[error("{0} must not be empty")]
    EmptyMedia(&'static str),
    #[error("media byte length exceeds the portable 2 GiB boundary")]
    MediaTooLarge,
    #[error("media type must be a bounded lowercase type/subtype token")]
    InvalidMediaType,
    #[error("3D file role and concrete format are incompatible")]
    InvalidFile3DRole,
    #[error("3D file bytes do not match the declared concrete format")]
    InvalidFile3DFormat,
    #[error("AUDIO tensor and sample rate do not match the source contract")]
    InvalidAudio,
    #[error("VIDEO tensor and frame rate do not match the source contract")]
    InvalidVideo,
    #[error("camera fields or dimensions do not match the source contract")]
    InvalidCamera,
    #[error("SPLAT tensors do not match the canonical Gaussian layout")]
    InvalidSplat,
    #[error("MESH tensors do not match the canonical indexed-triangle layout")]
    InvalidMesh,
    #[error("VOXEL tensors do not match the canonical density/color layout")]
    InvalidVoxel,
    #[error("payload resident-byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error("resident tensor allocation changed")]
    ResidentAllocationChanged,
    #[error("payload digest or resident-byte projection changed")]
    ProjectionChanged,
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    DescriptorEncoding(#[from] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum NativeCancellableMediaPayloadError {
    #[error("media payload construction was cancelled")]
    Cancelled,
    #[error(transparent)]
    Payload(#[from] NativeMediaPayloadError),
}

fn project_bounding_boxes(
    frames: &[Box<[NativeBoundingBox]>],
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    check_count("bounding box frames", frames.len(), MAX_FRAMES)?;
    let mut projection =
        Projection::new::<NativeBoundingBoxPayload>(b"zed.sim.media.bounding-box.v1")?;
    projection.hash_len(frames.len())?;
    projection.add_allocation::<Box<[NativeBoundingBox]>>(frames.len())?;
    for frame in frames {
        check_count("bounding boxes per frame", frame.len(), MAX_ITEMS)?;
        projection.hash_len(frame.len())?;
        projection.add_allocation::<NativeBoundingBox>(frame.len())?;
        for bounding_box in frame.iter() {
            bounding_box.validate()?;
            projection.hash_f64(bounding_box.x);
            projection.hash_f64(bounding_box.y);
            projection.hash_f64(bounding_box.width);
            projection.hash_f64(bounding_box.height);
            projection.hash_optional_text(bounding_box.label.as_deref())?;
            projection.hash_optional_f64(bounding_box.score);
        }
    }
    Ok(projection.finish())
}

fn project_face_landmarks(
    image_height: u32,
    image_width: u32,
    frames: &[Box<[NativeFaceLandmark]>],
    connection_sets: &[NativeFaceConnectionSet],
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    require_image_size(image_height, image_width)?;
    check_count("face landmark frames", frames.len(), MAX_FRAMES)?;
    check_count("face connection sets", connection_sets.len(), MAX_ITEMS)?;
    let mut projection =
        Projection::new::<NativeFaceLandmarksPayload>(b"zed.sim.media.face-landmarks.v1")?;
    projection.hasher.update(image_height.to_le_bytes());
    projection.hasher.update(image_width.to_le_bytes());
    projection.hash_len(frames.len())?;
    projection.add_allocation::<Box<[NativeFaceLandmark]>>(frames.len())?;
    for frame in frames {
        check_count("faces per frame", frame.len(), MAX_ITEMS)?;
        projection.hash_len(frame.len())?;
        projection.add_allocation::<NativeFaceLandmark>(frame.len())?;
        for face in frame.iter() {
            face.validate()?;
            for value in face.bbox_xyxy {
                projection.hash_f64(value);
            }
            projection.hash_len(face.blendshapes.len())?;
            projection.add_allocation::<NativeFaceBlendshape>(face.blendshapes.len())?;
            for blendshape in face.blendshapes.iter() {
                projection.hash_text(&blendshape.name)?;
                projection.hash_f64(blendshape.score);
            }
            projection.hash_len(face.landmarks_xy.len())?;
            projection.add_allocation::<NativePoint2>(face.landmarks_xy.len())?;
            for point in face.landmarks_xy.iter() {
                projection.hash_f64(point.x);
                projection.hash_f64(point.y);
            }
            projection.hash_len(face.landmarks_3d.len())?;
            projection.add_allocation::<NativePoint3>(face.landmarks_3d.len())?;
            for point in face.landmarks_3d.iter() {
                projection.hash_f64(point.x);
                projection.hash_f64(point.y);
                projection.hash_f64(point.z);
            }
            projection.hash_f64(face.presence);
            projection.hash_f64(face.score);
            for value in face.transformation_matrix {
                projection.hash_f64(value);
            }
        }
    }
    projection.hash_len(connection_sets.len())?;
    projection.add_allocation::<NativeFaceConnectionSet>(connection_sets.len())?;
    let mut names = BTreeSet::new();
    for connection_set in connection_sets {
        if !names.insert(connection_set.name()) {
            return Err(NativeMediaPayloadError::DuplicateConnectionSet);
        }
        projection.hash_text(&connection_set.name)?;
        projection.hash_len(connection_set.edges.len())?;
        projection.add_allocation::<[u16; 2]>(connection_set.edges.len())?;
        let mut previous = None;
        for edge in connection_set.edges.iter() {
            if usize::from(edge[0]) >= FACE_LANDMARK_COUNT
                || usize::from(edge[1]) >= FACE_LANDMARK_COUNT
                || edge[0] == edge[1]
            {
                return Err(NativeMediaPayloadError::InvalidFaceConnection(*edge));
            }
            if previous.is_some_and(|previous| previous >= *edge) {
                return Err(NativeMediaPayloadError::DuplicateFaceConnection);
            }
            projection.hasher.update(edge[0].to_le_bytes());
            projection.hasher.update(edge[1].to_le_bytes());
            previous = Some(*edge);
        }
    }
    Ok(projection.finish())
}

fn project_pose_keypoints(
    frames: &[NativePoseFrame],
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    check_count("pose frames", frames.len(), MAX_FRAMES)?;
    let mut projection =
        Projection::new::<NativePoseKeypointPayload>(b"zed.sim.media.pose-keypoint.v1")?;
    projection.hash_len(frames.len())?;
    projection.add_allocation::<NativePoseFrame>(frames.len())?;
    for frame in frames {
        require_image_size(frame.canvas_height, frame.canvas_width)?;
        projection.hasher.update(frame.canvas_width.to_le_bytes());
        projection.hasher.update(frame.canvas_height.to_le_bytes());
        projection.hash_len(frame.people.len())?;
        projection.add_allocation::<NativePosePerson>(frame.people.len())?;
        for person in frame.people.iter() {
            person.validate()?;
            for points in [
                person.pose.as_ref(),
                person.foot.as_ref(),
                person.face.as_ref(),
                person.hand_right.as_ref(),
                person.hand_left.as_ref(),
            ] {
                projection.hash_len(points.len())?;
                projection.add_allocation::<NativePoseKeypoint>(points.len())?;
                for point in points {
                    projection.hash_f64(point.x);
                    projection.hash_f64(point.y);
                    projection.hash_f64(point.score);
                }
            }
        }
    }
    Ok(projection.finish())
}

fn validate_tracks(
    track_path: &Tensor,
    track_visibility: &Tensor,
) -> Result<(), NativeMediaPayloadError> {
    let path = track_path.descriptor();
    let visibility = track_visibility.descriptor();
    let path_shape = path.shape();
    let visibility_shape = visibility.shape();
    if path.dtype() != DType::F32
        || visibility.dtype() != DType::Bool
        || path_shape.len() != 3
        || path_shape.get(2) != Some(&2)
        || visibility_shape.len() != 2
        || path_shape.get(0) != visibility_shape.first()
        || path_shape.get(1) != visibility_shape.get(1)
        || path.device() != visibility.device()
        || path.stream() != visibility.stream()
        || !path.is_contiguous()?
        || !visibility.is_contiguous()?
    {
        return Err(NativeMediaPayloadError::InvalidTracksShape);
    }
    track_path.contiguous_bytes()?;
    track_visibility.contiguous_bytes()?;
    Ok(())
}

fn project_tracks(
    track_path: &Tensor,
    track_visibility: &Tensor,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<NativeTracksPayload>(b"zed.sim.media.tracks.v1")?;
    projection.hash_tensor(b"track-path", track_path)?;
    projection.hash_tensor(b"track-visibility", track_visibility)?;
    projection.add_tensor_storages([track_path, track_visibility])?;
    Ok(projection.finish())
}

fn media_resident_parts<'a>(
    owned_bytes: u64,
    tensors: impl IntoIterator<Item = &'a Tensor>,
) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
    let mut storages = BTreeMap::new();
    for tensor in tensors {
        let storage_id = tensor.storage_id();
        let resident_bytes = tensor.storage_byte_len();
        if let Some(existing) = storages.insert(storage_id.get(), (storage_id, resident_bytes))
            && existing.1 != resident_bytes
        {
            return Err(NativeMediaPayloadError::ResidentAllocationChanged);
        }
    }
    let parts = NativeMediaResidentParts {
        owned_bytes,
        tensor_allocations: storages
            .into_values()
            .map(
                |(storage_id, resident_bytes)| NativeMediaTensorResidentAllocation {
                    storage_id,
                    resident_bytes,
                },
            )
            .collect(),
    };
    parts.resident_bytes()?;
    Ok(parts)
}

fn validate_sam3_track_data(
    packed_masks: Option<&Tensor>,
    frame_count: u64,
    scores: &[f64],
) -> Result<(), NativeMediaPayloadError> {
    check_count("SAM3 scores", scores.len(), MAX_ITEMS)?;
    for score in scores {
        require_probability("SAM3 object score", *score)?;
    }
    match packed_masks {
        Some(packed_masks) => {
            let descriptor = packed_masks.descriptor();
            let shape = descriptor.shape();
            let object_count = shape.get(1).copied();
            if descriptor.dtype() != DType::U8
                || shape.len() != 4
                || shape.first().copied() != Some(frame_count)
                || object_count.and_then(|count| usize::try_from(count).ok()) != Some(scores.len())
                || shape.get(2).copied() == Some(0)
                || shape.get(3).copied() == Some(0)
                || !descriptor.is_contiguous()?
            {
                return Err(NativeMediaPayloadError::InvalidSam3TrackData);
            }
            packed_masks.contiguous_bytes()?;
        }
        None if !scores.is_empty() => {
            return Err(NativeMediaPayloadError::InvalidSam3TrackData);
        }
        None => {}
    }
    Ok(())
}

fn project_sam3_track_data(
    packed_masks: Option<&Tensor>,
    frame_count: u64,
    scores: &[f64],
    original_height: u32,
    original_width: u32,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection =
        Projection::new::<NativeSam3TrackDataPayload>(b"zed.sim.media.sam3-track-data.v1")?;
    projection.hasher.update(frame_count.to_le_bytes());
    projection.hasher.update(original_height.to_le_bytes());
    projection.hasher.update(original_width.to_le_bytes());
    projection.hash_len(scores.len())?;
    projection.add_allocation::<f64>(scores.len())?;
    for score in scores {
        projection.hash_f64(*score);
    }
    match packed_masks {
        Some(packed_masks) => {
            projection.hasher.update([1]);
            projection.hash_tensor(b"packed-masks", packed_masks)?;
            projection.add_tensor_storages([packed_masks])?;
        }
        None => projection.hasher.update([0]),
    }
    Ok(projection.finish())
}

fn kind_tag(kind: NativeArtifactKind) -> u8 {
    match kind {
        NativeArtifactKind::AudioRecord => 0,
        NativeArtifactKind::Svg => 1,
        NativeArtifactKind::Webcam => 2,
    }
}

fn validate_file_3d_contents(
    format: NativeFile3DFormat,
    bytes: &[u8],
) -> Result<(), NativeMediaPayloadError> {
    let valid = match format {
        NativeFile3DFormat::Fbx => {
            bytes.starts_with(b"Kaydara FBX Binary") || bytes.starts_with(b"; FBX")
        }
        NativeFile3DFormat::Gltf => std::str::from_utf8(bytes)
            .ok()
            .is_some_and(|text| text.trim_start().starts_with('{')),
        NativeFile3DFormat::Glb => bytes.len() >= 12 && bytes.starts_with(b"glTF"),
        NativeFile3DFormat::Ksplat => bytes.len() >= 4096 + 1024 && bytes[0] == 0 && bytes[1] >= 1,
        NativeFile3DFormat::Obj => std::str::from_utf8(bytes).ok().is_some_and(|text| {
            text.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("v ") || line.starts_with("f ")
            })
        }),
        NativeFile3DFormat::Ply => bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r\n"),
        NativeFile3DFormat::Splat => bytes.len().is_multiple_of(32),
        NativeFile3DFormat::Spz => bytes.starts_with(&[0x1f, 0x8b]),
        NativeFile3DFormat::Stl => {
            bytes.starts_with(b"solid")
                || bytes.get(80..84).is_some_and(|count| {
                    let count = u32::from_le_bytes([count[0], count[1], count[2], count[3]]);
                    usize::try_from(count)
                        .ok()
                        .and_then(|count| count.checked_mul(50))
                        .and_then(|body| body.checked_add(84))
                        == Some(bytes.len())
                })
        }
        NativeFile3DFormat::Usdz => bytes.starts_with(b"PK\x03\x04"),
    };
    if valid {
        Ok(())
    } else {
        Err(NativeMediaPayloadError::InvalidFile3DFormat)
    }
}

fn file_format_tag(format: NativeFile3DFormat) -> u8 {
    match format {
        NativeFile3DFormat::Fbx => 0,
        NativeFile3DFormat::Gltf => 1,
        NativeFile3DFormat::Glb => 2,
        NativeFile3DFormat::Ksplat => 3,
        NativeFile3DFormat::Obj => 4,
        NativeFile3DFormat::Ply => 5,
        NativeFile3DFormat::Splat => 6,
        NativeFile3DFormat::Spz => 7,
        NativeFile3DFormat::Stl => 8,
        NativeFile3DFormat::Usdz => 9,
    }
}

fn file_role_tag(role: NativeFile3DRole) -> u8 {
    match role {
        NativeFile3DRole::Any => 0,
        NativeFile3DRole::PointCloudAny => 1,
        NativeFile3DRole::SplatAny => 2,
        NativeFile3DRole::Fbx => 3,
        NativeFile3DRole::Gltf => 4,
        NativeFile3DRole::Glb => 5,
        NativeFile3DRole::Ksplat => 6,
        NativeFile3DRole::Obj => 7,
        NativeFile3DRole::Ply => 8,
        NativeFile3DRole::Splat => 9,
        NativeFile3DRole::Spz => 10,
        NativeFile3DRole::Stl => 11,
        NativeFile3DRole::Usdz => 12,
    }
}

fn camera_role_tag(role: NativeCameraRole) -> u8 {
    match role {
        NativeCameraRole::CameraControl => 0,
        NativeCameraRole::Load3D => 1,
    }
}

fn check_byte_count(_field: &'static str, length: usize) -> Result<(), NativeMediaPayloadError> {
    if length > MAX_MEDIA_BYTES {
        return Err(NativeMediaPayloadError::MediaTooLarge);
    }
    Ok(())
}

fn checked_media_type(value: String) -> Result<String, NativeMediaPayloadError> {
    if value.is_empty()
        || value.len() > MAX_TEXT_BYTES
        || !value.contains('/')
        || value.bytes().any(|byte| {
            !(byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'/' | b'-' | b'+' | b'.'))
        })
    {
        return Err(NativeMediaPayloadError::InvalidMediaType);
    }
    Ok(value)
}

fn project_byte_asset<Payload>(
    domain: &[u8],
    tags: &[u8],
    media_type: &str,
    bytes: &[u8],
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<Payload>(domain)?;
    projection.hash_len(tags.len())?;
    projection.hasher.update(tags);
    projection.hash_text(media_type)?;
    projection.hash_len(bytes.len())?;
    projection.hasher.update(bytes);
    projection.add_bytes(bytes.len())?;
    Ok(projection.finish())
}

fn exact_tensor_parts<Payload, const COUNT: usize>(
    tensors: [&Tensor; COUNT],
    expected_resident_bytes: u64,
) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
    exact_tensor_parts_with_owned(mem::size_of::<Payload>(), tensors, expected_resident_bytes)
}

fn exact_tensor_parts_with_owned<'a>(
    owned_bytes: usize,
    tensors: impl IntoIterator<Item = &'a Tensor>,
    expected_resident_bytes: u64,
) -> Result<NativeMediaResidentParts, NativeMediaPayloadError> {
    let owned_bytes =
        u64::try_from(owned_bytes).map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?;
    let parts = media_resident_parts(owned_bytes, tensors)?;
    if parts.resident_bytes()? != expected_resident_bytes {
        return Err(NativeMediaPayloadError::ProjectionChanged);
    }
    Ok(parts)
}

fn validate_audio(waveform: &Tensor, sample_rate: u32) -> Result<(), NativeMediaPayloadError> {
    let descriptor = waveform.descriptor();
    let [batch, channels, _samples] = descriptor.shape() else {
        return Err(NativeMediaPayloadError::InvalidAudio);
    };
    if descriptor.dtype() != DType::F32
        || *batch == 0
        || *channels == 0
        || !(1..=384_000).contains(&sample_rate)
        || !descriptor
            .is_contiguous()
            .map_err(NativeMediaPayloadError::Tensor)?
    {
        return Err(NativeMediaPayloadError::InvalidAudio);
    }
    validate_finite_f32(waveform, NativeMediaPayloadError::InvalidAudio)
}

fn validate_audio_cancellable(
    waveform: &Tensor,
    sample_rate: u32,
    cancellation: &CancellationToken,
) -> Result<(), NativeCancellableMediaPayloadError> {
    let descriptor = waveform.descriptor();
    let [batch, channels, _samples] = descriptor.shape() else {
        return Err(NativeMediaPayloadError::InvalidAudio.into());
    };
    if descriptor.dtype() != DType::F32
        || *batch == 0
        || *channels == 0
        || !(1..=384_000).contains(&sample_rate)
        || !descriptor
            .is_contiguous()
            .map_err(NativeMediaPayloadError::Tensor)?
    {
        return Err(NativeMediaPayloadError::InvalidAudio.into());
    }
    validate_finite_f32_cancellable(
        waveform,
        NativeMediaPayloadError::InvalidAudio,
        cancellation,
    )
}

fn project_audio(
    waveform: &Tensor,
    sample_rate: u32,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<NativeAudioPayload>(b"zed.sim.media.audio.v1")?;
    projection.hasher.update(sample_rate.to_le_bytes());
    projection.hash_tensor(b"waveform", waveform)?;
    projection.add_tensor_storages([waveform])?;
    Ok(projection.finish())
}

fn project_audio_cancellable(
    waveform: &Tensor,
    sample_rate: u32,
    cancellation: &CancellationToken,
) -> Result<([u8; 32], u64), NativeCancellableMediaPayloadError> {
    let mut projection = Projection::new::<NativeAudioPayload>(b"zed.sim.media.audio.v1")?;
    projection.hasher.update(sample_rate.to_le_bytes());
    projection.hash_tensor_cancellable(b"waveform", waveform, cancellation)?;
    projection.add_tensor_storages([waveform])?;
    Ok(projection.finish())
}

fn validate_video(
    frames: &Tensor,
    frame_rate_numerator: u64,
    frame_rate_denominator: u64,
    _bit_depth: NativeVideoBitDepth,
    audio: Option<&NativeAudioPayload>,
    alpha: Option<&Tensor>,
    metadata: &BTreeMap<String, String>,
) -> Result<(), NativeMediaPayloadError> {
    let descriptor = frames.descriptor();
    let shape = descriptor.shape();
    if !matches!(descriptor.dtype(), DType::F32 | DType::U8)
        || shape.len() != 4
        || shape.contains(&0)
        || !matches!(shape[3], 1 | 3 | 4)
        || frame_rate_numerator == 0
        || frame_rate_denominator == 0
        || greatest_common_divisor(frame_rate_numerator, frame_rate_denominator) != 1
        || alpha.is_some_and(|alpha| {
            alpha.descriptor().dtype() != DType::F32
                || alpha.descriptor().shape() != [shape[0], shape[1], shape[2], 1]
        })
        || metadata.len() > 128
        || metadata.iter().any(|(key, value)| {
            key.is_empty()
                || key.len() > MAX_TEXT_BYTES
                || value.len() > MAX_TEXT_BYTES
                || key.chars().any(char::is_control)
                || value.chars().any(char::is_control)
        })
    {
        return Err(NativeMediaPayloadError::InvalidVideo);
    }
    if descriptor.dtype() == DType::F32 {
        validate_finite_f32(frames, NativeMediaPayloadError::InvalidVideo)?;
    }
    if let Some(audio) = audio {
        audio.validate()?;
        if audio.waveform().descriptor().shape()[0] != 1 {
            return Err(NativeMediaPayloadError::InvalidVideo);
        }
    }
    if let Some(alpha) = alpha {
        validate_finite_f32(alpha, NativeMediaPayloadError::InvalidVideo)?;
        let values = tensor_f32_values(alpha).map_err(|_| NativeMediaPayloadError::InvalidVideo)?;
        if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
            return Err(NativeMediaPayloadError::InvalidVideo);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_video_cancellable(
    frames: &Tensor,
    frame_rate_numerator: u64,
    frame_rate_denominator: u64,
    _bit_depth: NativeVideoBitDepth,
    audio: Option<&NativeAudioPayload>,
    alpha: Option<&Tensor>,
    metadata: &BTreeMap<String, String>,
    cancellation: &CancellationToken,
) -> Result<(), NativeCancellableMediaPayloadError> {
    check_media_cancellation(cancellation)?;
    let descriptor = frames.descriptor();
    let shape = descriptor.shape();
    if !matches!(descriptor.dtype(), DType::F32 | DType::U8)
        || shape.len() != 4
        || shape.contains(&0)
        || !matches!(shape[3], 1 | 3 | 4)
        || frame_rate_numerator == 0
        || frame_rate_denominator == 0
        || greatest_common_divisor(frame_rate_numerator, frame_rate_denominator) != 1
        || alpha.is_some_and(|alpha| {
            alpha.descriptor().dtype() != DType::F32
                || alpha.descriptor().shape() != [shape[0], shape[1], shape[2], 1]
        })
        || metadata.len() > 128
        || metadata.iter().any(|(key, value)| {
            key.is_empty()
                || key.len() > MAX_TEXT_BYTES
                || value.len() > MAX_TEXT_BYTES
                || key.chars().any(char::is_control)
                || value.chars().any(char::is_control)
        })
    {
        return Err(NativeMediaPayloadError::InvalidVideo.into());
    }
    if descriptor.dtype() == DType::F32 {
        validate_finite_f32_cancellable(
            frames,
            NativeMediaPayloadError::InvalidVideo,
            cancellation,
        )?;
    }
    if let Some(audio) = audio {
        validate_audio_cancellable(audio.waveform(), audio.sample_rate(), cancellation)?;
        if audio.waveform().descriptor().shape()[0] != 1 {
            return Err(NativeMediaPayloadError::InvalidVideo.into());
        }
        let (digest, resident_bytes) =
            project_audio_cancellable(audio.waveform(), audio.sample_rate(), cancellation)?;
        require_projection(
            *audio.semantic_digest_sha256(),
            digest,
            audio.resident_bytes(),
            resident_bytes,
        )?;
        audio.resident_parts()?;
    }
    if let Some(alpha) = alpha {
        let bytes = alpha
            .contiguous_bytes()
            .map_err(NativeMediaPayloadError::Tensor)?;
        if bytes.len() % 4 != 0 {
            return Err(NativeMediaPayloadError::InvalidVideo.into());
        }
        for chunk in bytes.chunks(64 * 1024) {
            check_media_cancellation(cancellation)?;
            if chunk.chunks_exact(4).any(|value| {
                let value = f32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                !value.is_finite() || !(0.0..=1.0).contains(&value)
            }) {
                return Err(NativeMediaPayloadError::InvalidVideo.into());
            }
        }
    }
    check_media_cancellation(cancellation)
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn project_video(
    frames: &Tensor,
    frame_rate_numerator: u64,
    frame_rate_denominator: u64,
    bit_depth: NativeVideoBitDepth,
    audio: Option<&NativeAudioPayload>,
    alpha: Option<&Tensor>,
    metadata: &BTreeMap<String, String>,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<NativeVideoPayload>(b"zed.sim.media.video.v2")?;
    projection.hasher.update(frame_rate_numerator.to_le_bytes());
    projection
        .hasher
        .update(frame_rate_denominator.to_le_bytes());
    projection.hasher.update([bit_depth.bits()]);
    projection.hash_tensor(b"frames", frames)?;
    let mut storages = vec![frames];
    match audio {
        Some(audio) => {
            projection.hasher.update([1]);
            projection.hasher.update(audio.semantic_digest_sha256());
            storages.push(audio.waveform());
        }
        None => projection.hasher.update([0]),
    }
    match alpha {
        Some(alpha) => {
            projection.hasher.update([1]);
            projection.hash_tensor(b"alpha", alpha)?;
            storages.push(alpha);
        }
        None => projection.hasher.update([0]),
    }
    projection.hash_len(metadata.len())?;
    for (key, value) in metadata {
        projection.hash_len(key.len())?;
        projection.hasher.update(key.as_bytes());
        projection.hash_len(value.len())?;
        projection.hasher.update(value.as_bytes());
        projection.add_bytes(key.capacity())?;
        projection.add_bytes(value.capacity())?;
    }
    projection.add_tensor_storages(storages)?;
    Ok(projection.finish())
}

#[allow(clippy::too_many_arguments)]
fn project_video_cancellable(
    frames: &Tensor,
    frame_rate_numerator: u64,
    frame_rate_denominator: u64,
    bit_depth: NativeVideoBitDepth,
    audio: Option<&NativeAudioPayload>,
    alpha: Option<&Tensor>,
    metadata: &BTreeMap<String, String>,
    cancellation: &CancellationToken,
) -> Result<([u8; 32], u64), NativeCancellableMediaPayloadError> {
    let mut projection = Projection::new::<NativeVideoPayload>(b"zed.sim.media.video.v2")?;
    projection.hasher.update(frame_rate_numerator.to_le_bytes());
    projection
        .hasher
        .update(frame_rate_denominator.to_le_bytes());
    projection.hasher.update([bit_depth.bits()]);
    projection.hash_tensor_cancellable(b"frames", frames, cancellation)?;
    let mut storages = vec![frames];
    match audio {
        Some(audio) => {
            projection.hasher.update([1]);
            projection.hasher.update(audio.semantic_digest_sha256());
            storages.push(audio.waveform());
        }
        None => projection.hasher.update([0]),
    }
    match alpha {
        Some(alpha) => {
            projection.hasher.update([1]);
            projection.hash_tensor_cancellable(b"alpha", alpha, cancellation)?;
            storages.push(alpha);
        }
        None => projection.hasher.update([0]),
    }
    projection.hash_len(metadata.len())?;
    for (key, value) in metadata {
        check_media_cancellation(cancellation)?;
        projection.hash_len(key.len())?;
        projection.hasher.update(key.as_bytes());
        projection.hash_len(value.len())?;
        projection.hasher.update(value.as_bytes());
        projection.add_bytes(key.capacity())?;
        projection.add_bytes(value.capacity())?;
    }
    projection.add_tensor_storages(storages)?;
    check_media_cancellation(cancellation)?;
    Ok(projection.finish())
}

fn validate_encoded_h264_mp4(
    bytes: &Tensor,
    content_sha256: [u8; 32],
    dimensions: (u64, u64),
    frame_rate: (u64, u64),
    frame_count: u64,
) -> Result<(), NativeMediaPayloadError> {
    let descriptor = bytes.descriptor();
    let [byte_count] = descriptor.shape() else {
        return Err(NativeMediaPayloadError::InvalidVideo);
    };
    if descriptor.dtype() != DType::U8
        || descriptor.device() != DeviceId::CPU
        || *byte_count == 0
        || !descriptor.is_contiguous()?
        || dimensions.0 == 0
        || dimensions.1 == 0
        || frame_rate.0 == 0
        || frame_rate.1 == 0
        || frame_rate.0 > i32::MAX as u64
        || frame_rate.1 > i32::MAX as u64
        || greatest_common_divisor(frame_rate.0, frame_rate.1) != 1
        || frame_count == 0
    {
        return Err(NativeMediaPayloadError::InvalidVideo);
    }
    let actual_content_sha256: [u8; 32] = Sha256::digest(bytes.contiguous_bytes()?).into();
    if actual_content_sha256 != content_sha256 {
        return Err(NativeMediaPayloadError::InvalidVideo);
    }
    Ok(())
}

fn project_encoded_h264_mp4(
    bytes: &Tensor,
    content_sha256: [u8; 32],
    source_video_sha256: [u8; 32],
    dimensions: (u64, u64),
    frame_rate: (u64, u64),
    frame_count: u64,
    bit_depth: NativeVideoBitDepth,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection =
        Projection::new::<NativeVideoPayload>(b"zed.sim.media.video.encoded-h264-mp4.v1")?;
    projection.hasher.update(b"mp4\0h264\0");
    match bit_depth {
        NativeVideoBitDepth::Eight => projection.hasher.update(b"yuv420p\0"),
        NativeVideoBitDepth::Ten => projection.hasher.update(b"yuv420p10le\0"),
    }
    projection.hasher.update([bit_depth.bits(), 0, 0]);
    projection.hasher.update(dimensions.0.to_le_bytes());
    projection.hasher.update(dimensions.1.to_le_bytes());
    projection.hasher.update(frame_rate.0.to_le_bytes());
    projection.hasher.update(frame_rate.1.to_le_bytes());
    projection.hasher.update(frame_count.to_le_bytes());
    projection.hasher.update(source_video_sha256);
    projection.hasher.update(content_sha256);
    projection.hash_tensor(b"encoded-bytes", bytes)?;
    projection.add_tensor_storages([bytes])?;
    Ok(projection.finish())
}

fn validate_camera(
    position: &[f32; 3],
    target: &[f32; 3],
    zoom: f32,
    orientation_wxyz: Option<&[f32; 4]>,
    projection: NativeCameraProjection,
    width: u32,
    height: u32,
) -> Result<(), NativeMediaPayloadError> {
    if width == 0
        || height == 0
        || !(0.01..=100.0).contains(&zoom)
        || position
            .iter()
            .chain(target)
            .any(|value| !value.is_finite())
        || orientation_wxyz.is_some_and(|orientation| {
            orientation.iter().any(|value| !value.is_finite())
                || (orientation
                    .iter()
                    .fold(0.0_f32, |sum, value| value.mul_add(*value, sum))
                    - 1.0)
                    .abs()
                    > 1.0e-3
        })
        || orientation_wxyz.is_none()
            && position
                .iter()
                .zip(target)
                .all(|(position, target)| position == target)
        || match projection {
            NativeCameraProjection::Perspective {
                fov_degrees,
                aspect_ratio,
                near,
                far,
            } => {
                !(1.0..=120.0).contains(&fov_degrees)
                    || !aspect_ratio.is_finite()
                    || aspect_ratio <= 0.0
                    || !near.is_finite()
                    || near <= 0.0
                    || !far.is_finite()
                    || far <= near
            }
            NativeCameraProjection::Orthographic {
                left,
                right,
                bottom,
                top,
                near,
                far,
            } => {
                [left, right, bottom, top, near, far]
                    .iter()
                    .any(|value| !value.is_finite())
                    || left >= right
                    || bottom >= top
                    || near <= 0.0
                    || far <= near
            }
        }
    {
        return Err(NativeMediaPayloadError::InvalidCamera);
    }
    Ok(())
}

fn project_camera<Payload>(
    role: NativeCameraRole,
    position: &[f32; 3],
    target: &[f32; 3],
    zoom: f32,
    orientation_wxyz: Option<&[f32; 4]>,
    camera_projection: NativeCameraProjection,
    width: u32,
    height: u32,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<Payload>(b"zed.sim.media.camera.v1")?;
    projection.hasher.update([camera_role_tag(role)]);
    for value in position.iter().chain(target).chain(std::iter::once(&zoom)) {
        projection.hasher.update(value.to_bits().to_le_bytes());
    }
    match orientation_wxyz {
        Some(orientation) => {
            projection.hasher.update([1]);
            for value in orientation {
                projection.hasher.update(value.to_bits().to_le_bytes());
            }
        }
        None => projection.hasher.update([0]),
    }
    match camera_projection {
        NativeCameraProjection::Perspective {
            fov_degrees,
            aspect_ratio,
            near,
            far,
        } => {
            projection.hasher.update([0]);
            for value in [fov_degrees, aspect_ratio, near, far] {
                projection.hasher.update(value.to_bits().to_le_bytes());
            }
        }
        NativeCameraProjection::Orthographic {
            left,
            right,
            bottom,
            top,
            near,
            far,
        } => {
            projection.hasher.update([1]);
            for value in [left, right, bottom, top, near, far] {
                projection.hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
    projection.hasher.update(width.to_le_bytes());
    projection.hasher.update(height.to_le_bytes());
    Ok(projection.finish())
}

fn validate_splat(
    positions: &Tensor,
    scales: &Tensor,
    rotations: &Tensor,
    opacity: &Tensor,
    spherical_harmonics: &Tensor,
    counts: Option<&[u64]>,
) -> Result<(), NativeMediaPayloadError> {
    let positions_shape = positions.descriptor().shape();
    let batch_and_count = positions_shape.get(..2);
    if positions.descriptor().dtype() != DType::F32
        || positions_shape.len() != 3
        || positions_shape[2] != 3
        || positions_shape[0] == 0
        || positions_shape[1] == 0
        || scales.descriptor().dtype() != DType::F32
        || scales.descriptor().shape() != [positions_shape[0], positions_shape[1], 3]
        || rotations.descriptor().dtype() != DType::F32
        || rotations.descriptor().shape() != [positions_shape[0], positions_shape[1], 4]
        || opacity.descriptor().dtype() != DType::F32
        || opacity.descriptor().shape() != [positions_shape[0], positions_shape[1], 1]
        || spherical_harmonics.descriptor().dtype() != DType::F32
        || spherical_harmonics.descriptor().shape().len() != 4
        || spherical_harmonics.descriptor().shape().get(..2) != batch_and_count
        || spherical_harmonics.descriptor().shape()[2] == 0
        || spherical_harmonics.descriptor().shape()[2] > MAX_SPLAT_SH_COEFFICIENTS
        || !matches!(spherical_harmonics.descriptor().shape()[2], 1 | 4 | 9 | 16)
        || spherical_harmonics.descriptor().shape()[3] != 3
        || counts.is_some_and(|counts| {
            counts.len() != usize::try_from(positions_shape[0]).unwrap_or(usize::MAX)
                || counts.iter().any(|count| *count > positions_shape[1])
        })
    {
        return Err(NativeMediaPayloadError::InvalidSplat);
    }
    for tensor in [positions, scales, rotations, opacity, spherical_harmonics] {
        validate_finite_f32(tensor, NativeMediaPayloadError::InvalidSplat)?;
    }
    validate_splat_values(
        positions,
        scales,
        rotations,
        opacity,
        spherical_harmonics,
        counts,
    )
}

fn validate_splat_values(
    positions: &Tensor,
    scales: &Tensor,
    rotations: &Tensor,
    opacity: &Tensor,
    spherical_harmonics: &Tensor,
    counts: Option<&[u64]>,
) -> Result<(), NativeMediaPayloadError> {
    let batch_count = usize::try_from(positions.descriptor().shape()[0])
        .map_err(|_| NativeMediaPayloadError::InvalidSplat)?;
    let padded_count = usize::try_from(positions.descriptor().shape()[1])
        .map_err(|_| NativeMediaPayloadError::InvalidSplat)?;
    let coefficient_count = usize::try_from(spherical_harmonics.descriptor().shape()[2])
        .map_err(|_| NativeMediaPayloadError::InvalidSplat)?;
    let positions = tensor_f32_values(positions)?;
    let scales = tensor_f32_values(scales)?;
    let rotations = tensor_f32_values(rotations)?;
    let opacity = tensor_f32_values(opacity)?;
    let spherical_harmonics = tensor_f32_values(spherical_harmonics)?;
    for batch in 0..batch_count {
        let active_count = counts
            .and_then(|counts| counts.get(batch).copied())
            .map_or(Ok(padded_count), |count| {
                usize::try_from(count).map_err(|_| NativeMediaPayloadError::InvalidSplat)
            })?;
        for item in 0..padded_count {
            let linear = batch
                .checked_mul(padded_count)
                .and_then(|value| value.checked_add(item))
                .ok_or(NativeMediaPayloadError::InvalidSplat)?;
            let position = tensor_item(&positions, linear, 3)?;
            let scale = tensor_item(&scales, linear, 3)?;
            let rotation = tensor_item(&rotations, linear, 4)?;
            let alpha = *opacity
                .get(linear)
                .ok_or(NativeMediaPayloadError::InvalidSplat)?;
            let harmonics = tensor_item(
                &spherical_harmonics,
                linear,
                coefficient_count
                    .checked_mul(3)
                    .ok_or(NativeMediaPayloadError::InvalidSplat)?,
            )?;
            if item < active_count {
                if scale.iter().any(|value| *value <= 0.0) || !(0.0..=1.0).contains(&alpha) {
                    return Err(NativeMediaPayloadError::InvalidSplat);
                }
                let norm_squared = rotation
                    .iter()
                    .try_fold(0.0_f32, |sum, value| {
                        let next = value.mul_add(*value, sum);
                        next.is_finite().then_some(next)
                    })
                    .ok_or(NativeMediaPayloadError::InvalidSplat)?;
                if (norm_squared - 1.0).abs() > 1.0e-3 {
                    return Err(NativeMediaPayloadError::InvalidSplat);
                }
            } else if position
                .iter()
                .chain(scale)
                .chain(rotation)
                .chain(std::iter::once(&alpha))
                .chain(harmonics)
                .any(|value| *value != 0.0)
            {
                return Err(NativeMediaPayloadError::InvalidSplat);
            }
        }
    }
    Ok(())
}

fn tensor_f32_values(tensor: &Tensor) -> Result<Vec<f32>, NativeMediaPayloadError> {
    let bytes = tensor
        .contiguous_bytes()
        .map_err(NativeMediaPayloadError::Tensor)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(NativeMediaPayloadError::InvalidSplat);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn tensor_item(
    values: &[f32],
    index: usize,
    width: usize,
) -> Result<&[f32], NativeMediaPayloadError> {
    let start = index
        .checked_mul(width)
        .ok_or(NativeMediaPayloadError::InvalidSplat)?;
    let end = start
        .checked_add(width)
        .ok_or(NativeMediaPayloadError::InvalidSplat)?;
    values
        .get(start..end)
        .ok_or(NativeMediaPayloadError::InvalidSplat)
}

fn project_splat<Payload>(
    positions: &Tensor,
    scales: &Tensor,
    rotations: &Tensor,
    opacity: &Tensor,
    spherical_harmonics: &Tensor,
    counts: Option<&[u64]>,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<Payload>(b"zed.sim.media.splat.v1")?;
    for (role, tensor) in [
        (&b"positions"[..], positions),
        (&b"scales"[..], scales),
        (&b"rotations"[..], rotations),
        (&b"opacity"[..], opacity),
        (&b"spherical-harmonics"[..], spherical_harmonics),
    ] {
        projection.hash_tensor(role, tensor)?;
    }
    projection.add_tensor_storages([positions, scales, rotations, opacity, spherical_harmonics])?;
    match counts {
        Some(counts) => {
            projection.hasher.update([1]);
            projection.hash_len(counts.len())?;
            projection.add_allocation::<u64>(counts.len())?;
            for count in counts {
                projection.hasher.update(count.to_le_bytes());
            }
        }
        None => projection.hasher.update([0]),
    }
    Ok(projection.finish())
}

fn validate_mesh_batch(
    vertices: &Tensor,
    faces: &Tensor,
    normals: Option<&Tensor>,
    uvs: Option<&Tensor>,
    colors: Option<&Tensor>,
    texture: Option<&Tensor>,
) -> Result<(), NativeMediaPayloadError> {
    let vertices_shape = vertices.descriptor().shape();
    let faces_shape = faces.descriptor().shape();
    if vertices.descriptor().dtype() != DType::F32
        || vertices_shape.len() != 2
        || vertices_shape[0] == 0
        || vertices_shape[1] != 3
        || !matches!(faces.descriptor().dtype(), DType::I32 | DType::I64)
        || faces_shape.len() != 2
        || faces_shape[0] == 0
        || faces_shape[1] != 3
        || normals.is_some_and(|tensor| {
            tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().shape() != vertices_shape
        })
        || uvs.is_some_and(|tensor| {
            tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().shape() != [vertices_shape[0], 2]
        })
        || colors.is_some_and(|tensor| {
            tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().shape().len() != 2
                || tensor.descriptor().shape()[0] != vertices_shape[0]
                || !matches!(tensor.descriptor().shape()[1], 3 | 4)
        })
        || texture.is_some_and(|tensor| {
            tensor.descriptor().dtype() != DType::F32
                || tensor.descriptor().shape().len() != 3
                || tensor.descriptor().shape()[0] == 0
                || tensor.descriptor().shape()[1] == 0
                || tensor.descriptor().shape()[2] != 3
        })
    {
        return Err(NativeMediaPayloadError::InvalidMesh);
    }
    validate_finite_f32(vertices, NativeMediaPayloadError::InvalidMesh)?;
    if let Some(normals) = normals {
        validate_finite_f32(normals, NativeMediaPayloadError::InvalidMesh)?;
    }
    if let Some(uvs) = uvs {
        validate_finite_f32(uvs, NativeMediaPayloadError::InvalidMesh)?;
    }
    if let Some(colors) = colors {
        validate_finite_f32(colors, NativeMediaPayloadError::InvalidMesh)?;
    }
    if let Some(texture) = texture {
        validate_finite_f32(texture, NativeMediaPayloadError::InvalidMesh)?;
    }
    validate_face_indices(faces, vertices_shape[0])
}

fn validate_face_indices(faces: &Tensor, vertex_count: u64) -> Result<(), NativeMediaPayloadError> {
    let bytes = faces.contiguous_bytes()?;
    match faces.descriptor().dtype() {
        DType::I32 => {
            for chunk in bytes.chunks_exact(4) {
                let index = i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if index < 0
                    || u64::try_from(index)
                        .ok()
                        .is_none_or(|index| index >= vertex_count)
                {
                    return Err(NativeMediaPayloadError::InvalidMesh);
                }
            }
        }
        DType::I64 => {
            for chunk in bytes.chunks_exact(8) {
                let index = i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                if index < 0
                    || u64::try_from(index)
                        .ok()
                        .is_none_or(|index| index >= vertex_count)
                {
                    return Err(NativeMediaPayloadError::InvalidMesh);
                }
            }
        }
        _ => return Err(NativeMediaPayloadError::InvalidMesh),
    }
    Ok(())
}

fn project_mesh<Payload>(
    batches: &[NativeMeshBatch],
    unlit: bool,
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<Payload>(b"zed.sim.media.mesh.v1")?;
    projection.hash_len(batches.len())?;
    projection.hasher.update([u8::from(unlit)]);
    projection.add_allocation::<NativeMeshBatch>(batches.len())?;
    let mut storages = Vec::new();
    for (index, batch) in batches.iter().enumerate() {
        projection.hasher.update(
            u64::try_from(index)
                .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?
                .to_le_bytes(),
        );
        projection.hash_tensor(b"vertices", &batch.vertices)?;
        projection.hash_tensor(b"faces", &batch.faces)?;
        storages.push(&batch.vertices);
        storages.push(&batch.faces);
        match &batch.normals {
            Some(normals) => {
                projection.hasher.update([1]);
                projection.hash_tensor(b"normals", normals)?;
                storages.push(normals);
            }
            None => projection.hasher.update([0]),
        }
        match &batch.colors {
            Some(colors) => {
                projection.hasher.update([1]);
                projection.hash_tensor(b"colors", colors)?;
                storages.push(colors);
            }
            None => projection.hasher.update([0]),
        }
        match &batch.uvs {
            Some(uvs) => {
                projection.hasher.update([1]);
                projection.hash_tensor(b"uvs", uvs)?;
                storages.push(uvs);
            }
            None => projection.hasher.update([0]),
        }
        match &batch.texture {
            Some(texture) => {
                projection.hasher.update([1]);
                projection.hash_tensor(b"texture", texture)?;
                storages.push(texture);
            }
            None => projection.hasher.update([0]),
        }
    }
    projection.add_tensor_storages(storages)?;
    Ok(projection.finish())
}

fn validate_voxel(
    density: &Tensor,
    colors: Option<&Tensor>,
    world_from_grid: &[f32; 16],
) -> Result<(), NativeMediaPayloadError> {
    let density_shape = density.descriptor().shape();
    if density.descriptor().dtype() != DType::F32
        || density_shape.len() != 4
        || density_shape.contains(&0)
        || colors.is_some_and(|colors| {
            colors.descriptor().dtype() != DType::F32
                || colors.descriptor().shape()
                    != [
                        density_shape[0],
                        density_shape[1],
                        density_shape[2],
                        density_shape[3],
                        3,
                    ]
        })
        || world_from_grid.iter().any(|value| !value.is_finite())
    {
        return Err(NativeMediaPayloadError::InvalidVoxel);
    }
    validate_finite_f32(density, NativeMediaPayloadError::InvalidVoxel)?;
    if let Some(colors) = colors {
        validate_finite_f32(colors, NativeMediaPayloadError::InvalidVoxel)?;
    }
    Ok(())
}

fn project_voxel<Payload>(
    density: &Tensor,
    colors: Option<&Tensor>,
    world_from_grid: &[f32; 16],
) -> Result<([u8; 32], u64), NativeMediaPayloadError> {
    let mut projection = Projection::new::<Payload>(b"zed.sim.media.voxel.v1")?;
    for value in world_from_grid {
        projection.hasher.update(value.to_bits().to_le_bytes());
    }
    projection.hash_tensor(b"density", density)?;
    let mut storages = vec![density];
    match colors {
        Some(colors) => {
            projection.hasher.update([1]);
            projection.hash_tensor(b"colors", colors)?;
            storages.push(colors);
        }
        None => projection.hasher.update([0]),
    }
    projection.add_tensor_storages(storages)?;
    Ok(projection.finish())
}

fn validate_finite_f32(
    tensor: &Tensor,
    error: NativeMediaPayloadError,
) -> Result<(), NativeMediaPayloadError> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(error);
    }
    let bytes = tensor
        .contiguous_bytes()
        .map_err(NativeMediaPayloadError::Tensor)?;
    if bytes.len() % 4 != 0
        || bytes
            .chunks_exact(4)
            .any(|chunk| !f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).is_finite())
    {
        return Err(error);
    }
    Ok(())
}

fn validate_finite_f32_cancellable(
    tensor: &Tensor,
    error: NativeMediaPayloadError,
    cancellation: &CancellationToken,
) -> Result<(), NativeCancellableMediaPayloadError> {
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(error.into());
    }
    let bytes = tensor
        .contiguous_bytes()
        .map_err(NativeMediaPayloadError::Tensor)?;
    if bytes.len() % 4 != 0 {
        return Err(error.into());
    }
    for chunk in bytes.chunks(64 * 1024) {
        check_media_cancellation(cancellation)?;
        if chunk
            .chunks_exact(4)
            .any(|value| !f32::from_ne_bytes([value[0], value[1], value[2], value[3]]).is_finite())
        {
            return Err(error.into());
        }
    }
    check_media_cancellation(cancellation)
}

#[cfg(test)]
thread_local! {
    static MEDIA_TEST_CANCEL_AFTER_CHECKS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
fn set_media_test_cancel_after_checks(checks: Option<usize>) {
    MEDIA_TEST_CANCEL_AFTER_CHECKS.set(checks);
}

fn check_media_cancellation(
    cancellation: &CancellationToken,
) -> Result<(), NativeCancellableMediaPayloadError> {
    #[cfg(test)]
    MEDIA_TEST_CANCEL_AFTER_CHECKS.with(|remaining| {
        if let Some(checks) = remaining.get() {
            if checks == 0 {
                remaining.set(None);
                cancellation.cancel();
            } else {
                remaining.set(Some(checks - 1));
            }
        }
    });
    cancellation
        .check()
        .map_err(|_| NativeCancellableMediaPayloadError::Cancelled)
}

struct Projection {
    hasher: Sha256,
    resident_bytes: u64,
}

impl Projection {
    fn new<Payload>(domain: &[u8]) -> Result<Self, NativeMediaPayloadError> {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update([0]);
        Ok(Self {
            hasher,
            resident_bytes: u64::try_from(mem::size_of::<Payload>())
                .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?,
        })
    }

    fn hash_len(&mut self, length: usize) -> Result<(), NativeMediaPayloadError> {
        self.hasher.update(
            u64::try_from(length)
                .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?
                .to_le_bytes(),
        );
        Ok(())
    }

    fn hash_f64(&mut self, value: f64) {
        self.hasher.update(value.to_bits().to_le_bytes());
    }

    fn hash_optional_f64(&mut self, value: Option<f64>) {
        match value {
            Some(value) => {
                self.hasher.update([1]);
                self.hash_f64(value);
            }
            None => self.hasher.update([0]),
        }
    }

    fn hash_text(&mut self, value: &str) -> Result<(), NativeMediaPayloadError> {
        self.hash_len(value.len())?;
        self.hasher.update(value.as_bytes());
        self.add_bytes(value.len())
    }

    fn hash_optional_text(&mut self, value: Option<&str>) -> Result<(), NativeMediaPayloadError> {
        match value {
            Some(value) => {
                self.hasher.update([1]);
                self.hash_text(value)
            }
            None => {
                self.hasher.update([0]);
                Ok(())
            }
        }
    }

    fn hash_tensor(&mut self, role: &[u8], tensor: &Tensor) -> Result<(), NativeMediaPayloadError> {
        self.hash_len(role.len())?;
        self.hasher.update(role);
        let descriptor = tensor.descriptor();
        let dtype = descriptor.dtype().catalog_name().as_bytes();
        self.hash_len(dtype.len())?;
        self.hasher.update(dtype);
        self.hash_len(descriptor.shape().len())?;
        for dimension in descriptor.shape() {
            self.hasher.update(dimension.to_le_bytes());
        }
        let bytes = tensor
            .contiguous_bytes()
            .map_err(NativeMediaPayloadError::Tensor)?;
        self.hash_len(bytes.len())?;
        self.hasher.update(bytes);
        Ok(())
    }

    fn hash_tensor_cancellable(
        &mut self,
        role: &[u8],
        tensor: &Tensor,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeCancellableMediaPayloadError> {
        check_media_cancellation(cancellation)?;
        self.hash_len(role.len())?;
        self.hasher.update(role);
        let descriptor = tensor.descriptor();
        let dtype = descriptor.dtype().catalog_name().as_bytes();
        self.hash_len(dtype.len())?;
        self.hasher.update(dtype);
        self.hash_len(descriptor.shape().len())?;
        for dimension in descriptor.shape() {
            self.hasher.update(dimension.to_le_bytes());
        }
        let bytes = tensor
            .contiguous_bytes()
            .map_err(NativeMediaPayloadError::Tensor)?;
        self.hash_len(bytes.len())?;
        for chunk in bytes.chunks(64 * 1024) {
            check_media_cancellation(cancellation)?;
            self.hasher.update(chunk);
        }
        check_media_cancellation(cancellation)
    }

    fn add_allocation<Value>(&mut self, length: usize) -> Result<(), NativeMediaPayloadError> {
        let bytes = mem::size_of::<Value>()
            .checked_mul(length)
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        self.add_bytes(bytes)
    }

    fn add_bytes(&mut self, bytes: usize) -> Result<(), NativeMediaPayloadError> {
        let bytes =
            u64::try_from(bytes).map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?;
        self.resident_bytes = self
            .resident_bytes
            .checked_add(bytes)
            .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
        Ok(())
    }

    fn add_tensor_storages<'a>(
        &mut self,
        tensors: impl IntoIterator<Item = &'a Tensor>,
    ) -> Result<(), NativeMediaPayloadError> {
        let mut seen = BTreeSet::<u64>::new();
        for tensor in tensors {
            let storage_id: StorageId = tensor.storage_id();
            if seen.insert(storage_id.get()) {
                self.resident_bytes = self
                    .resident_bytes
                    .checked_add(tensor.storage_byte_len())
                    .ok_or(NativeMediaPayloadError::ResidentBytesOverflow)?;
            }
        }
        Ok(())
    }

    fn finish(self) -> ([u8; 32], u64) {
        (self.hasher.finalize().into(), self.resident_bytes)
    }
}

fn require_projection(
    actual_digest: [u8; 32],
    expected_digest: [u8; 32],
    actual_resident_bytes: u64,
    expected_resident_bytes: u64,
) -> Result<(), NativeMediaPayloadError> {
    if actual_digest != expected_digest || actual_resident_bytes != expected_resident_bytes {
        return Err(NativeMediaPayloadError::ProjectionChanged);
    }
    Ok(())
}

fn require_finite(field: &'static str, value: f64) -> Result<(), NativeMediaPayloadError> {
    if !value.is_finite() {
        return Err(NativeMediaPayloadError::NonFinite { field });
    }
    Ok(())
}

fn require_non_negative(field: &'static str, value: f64) -> Result<(), NativeMediaPayloadError> {
    require_finite(field, value)?;
    if value < 0.0 {
        return Err(NativeMediaPayloadError::Negative(field));
    }
    Ok(())
}

fn require_probability(field: &'static str, value: f64) -> Result<(), NativeMediaPayloadError> {
    require_finite(field, value)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(NativeMediaPayloadError::NotProbability(field));
    }
    Ok(())
}

fn check_count(
    field: &'static str,
    actual: usize,
    maximum: usize,
) -> Result<(), NativeMediaPayloadError> {
    if actual > maximum {
        return Err(NativeMediaPayloadError::TooMany {
            field,
            actual,
            maximum,
        });
    }
    Ok(())
}

fn check_exact_count(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), NativeMediaPayloadError> {
    if actual != expected {
        return Err(NativeMediaPayloadError::WrongCardinality {
            field,
            actual,
            expected,
        });
    }
    Ok(())
}

fn checked_text(field: &'static str, value: String) -> Result<String, NativeMediaPayloadError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(NativeMediaPayloadError::InvalidText(field));
    }
    Ok(value)
}

fn require_image_size(height: u32, width: u32) -> Result<(), NativeMediaPayloadError> {
    if height == 0 || width == 0 {
        return Err(NativeMediaPayloadError::InvalidImageSize("image"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, DeviceId, ExecutionContext, StreamId,
        TensorDescriptor,
    };
    use std::{error::Error, mem, sync::Arc};

    fn tensor_on_stream(
        shape: Vec<u64>,
        dtype: DType,
        bytes: Vec<u8>,
        stream: StreamId,
    ) -> Result<Tensor, Box<dyn Error>> {
        let descriptor = TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, stream)?;
        let byte_count = u64::try_from(bytes.len())?;
        let (backend, authority) =
            CpuWorkspaceAuthority::create_backend(byte_count.saturating_add(64))?;
        let cancellation = CancellationToken::default();
        let context = ExecutionContext {
            stream,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let (tensor, _) = backend.upload_bytes(descriptor, &bytes, &context)?;
        Ok(tensor)
    }

    fn tensor(shape: Vec<u64>, dtype: DType, bytes: Vec<u8>) -> Result<Tensor, Box<dyn Error>> {
        tensor_on_stream(shape, dtype, bytes, StreamId::DEFAULT)
    }

    fn keypoints(count: usize) -> Result<Vec<NativePoseKeypoint>, NativeMediaPayloadError> {
        (0..count)
            .map(|index| {
                let coordinate = u32::try_from(index)
                    .map(f64::from)
                    .map_err(|_| NativeMediaPayloadError::ResidentBytesOverflow)?;
                NativePoseKeypoint::checked(coordinate, coordinate + 0.5, 0.75)
            })
            .collect()
    }

    fn pose_person() -> Result<NativePosePerson, NativeMediaPayloadError> {
        NativePosePerson::checked(
            keypoints(18)?,
            keypoints(6)?,
            keypoints(70)?,
            keypoints(21)?,
            keypoints(21)?,
        )
    }

    fn face() -> Result<NativeFaceLandmark, NativeMediaPayloadError> {
        let blendshapes = MEDIAPIPE_FACE_BLENDSHAPE_NAMES
            .iter()
            .map(|name| NativeFaceBlendshape::checked((*name).to_owned(), 0.25))
            .collect::<Result<Vec<_>, _>>()?;
        let point_2d = NativePoint2::checked(12.0, 24.0)?;
        let point_3d = NativePoint3::checked(1.0, 2.0, 3.0)?;
        NativeFaceLandmark::checked(
            [10.0, 20.0, 30.0, 40.0],
            blendshapes,
            vec![point_2d; FACE_LANDMARK_COUNT],
            vec![point_3d; FACE_LANDMARK_COUNT],
            -0.5,
            0.9,
            [0.0; 16],
        )
    }

    #[test]
    fn bounding_boxes_preserve_nested_source_shape_and_exact_accounting()
    -> Result<(), Box<dyn Error>> {
        let first =
            NativeBoundingBox::checked(-2.0, 3.0, 10.0, 20.0, Some("face".to_owned()), Some(0.75))?;
        let second = NativeBoundingBox::checked(0.0, 0.0, 0.0, 0.0, None, None)?;
        let payload = NativeBoundingBoxPayload::checked(vec![vec![first], vec![second]])?;

        payload.validate()?;
        assert_eq!(payload.frames().len(), 2);
        assert_eq!(payload.frames()[0][0].label(), Some("face"));
        let expected = u64::try_from(mem::size_of::<NativeBoundingBoxPayload>())?
            + u64::try_from(2 * mem::size_of::<Box<[NativeBoundingBox]>>())?
            + u64::try_from(2 * mem::size_of::<NativeBoundingBox>())?
            + 4;
        assert_eq!(payload.resident_bytes(), expected);

        let changed = NativeBoundingBoxPayload::checked(vec![vec![NativeBoundingBox::checked(
            -2.0,
            3.0,
            11.0,
            20.0,
            Some("face".to_owned()),
            Some(0.75),
        )?]])?;
        assert_ne!(
            payload.semantic_digest_sha256(),
            changed.semantic_digest_sha256()
        );
        assert!(NativeBoundingBox::checked(0.0, 0.0, -1.0, 1.0, None, None).is_err());
        assert!(NativeBoundingBox::checked(0.0, 0.0, 1.0, 1.0, None, Some(1.1)).is_err());
        Ok(())
    }

    #[test]
    fn face_landmarks_enforce_mediapipe_cardinalities_and_topology() -> Result<(), Box<dyn Error>> {
        let connection =
            NativeFaceConnectionSet::checked("face_oval".to_owned(), vec![[10, 20], [0, 1]])?;
        let payload =
            NativeFaceLandmarksPayload::checked(480, 640, vec![vec![face()?]], vec![connection])?;

        payload.validate()?;
        assert_eq!(payload.image_height(), 480);
        assert_eq!(payload.frames()[0][0].landmarks_xy().len(), 478);
        assert_eq!(payload.frames()[0][0].blendshapes().len(), 52);
        assert_eq!(payload.connection_sets()[0].edges(), &[[0, 1], [10, 20]]);
        assert!(
            payload.resident_bytes() > u64::try_from(mem::size_of::<NativeFaceLandmarksPayload>())?
        );

        let mut wrong_blendshapes = MEDIAPIPE_FACE_BLENDSHAPE_NAMES
            .iter()
            .map(|name| NativeFaceBlendshape::checked((*name).to_owned(), 0.0))
            .collect::<Result<Vec<_>, _>>()?;
        wrong_blendshapes[0] = NativeFaceBlendshape::checked("neutral".to_owned(), 0.0)?;
        assert!(
            NativeFaceLandmark::checked(
                [0.0, 0.0, 1.0, 1.0],
                wrong_blendshapes,
                vec![NativePoint2::checked(0.0, 0.0)?; FACE_LANDMARK_COUNT],
                vec![NativePoint3::checked(0.0, 0.0, 0.0)?; FACE_LANDMARK_COUNT],
                0.0,
                1.0,
                [0.0; 16],
            )
            .is_err()
        );
        assert!(NativeFaceConnectionSet::checked("all".to_owned(), vec![[1, 1]]).is_err());
        assert!(NativeFaceLandmarksPayload::checked(0, 640, vec![], vec![]).is_err());
        Ok(())
    }

    #[test]
    fn pose_keypoints_enforce_openpose_frame_shape() -> Result<(), Box<dyn Error>> {
        let frame = NativePoseFrame::checked(1920, 1080, vec![pose_person()?])?;
        let payload = NativePoseKeypointPayload::checked(vec![frame])?;

        payload.validate()?;
        let person = &payload.frames()[0].people()[0];
        assert_eq!(person.pose().len(), 18);
        assert_eq!(person.foot().len(), 6);
        assert_eq!(person.face().len(), 70);
        assert_eq!(person.hand_right().len(), 21);
        assert_eq!(person.hand_left().len(), 21);
        assert!(
            NativePosePerson::checked(
                keypoints(17)?,
                keypoints(6)?,
                keypoints(70)?,
                keypoints(21)?,
                keypoints(21)?,
            )
            .is_err()
        );
        assert!(NativePoseFrame::checked(0, 1080, vec![]).is_err());
        assert!(NativePoseKeypoint::checked(0.0, 0.0, f64::NAN).is_err());
        assert!(NativePoseKeypoint::checked(0.0, 0.0, -0.25).is_ok());
        assert!(NativePoseKeypoint::checked(0.0, 0.0, 1.25).is_ok());
        Ok(())
    }

    #[test]
    fn tracks_require_exact_tensor_roles_and_account_storage() -> Result<(), Box<dyn Error>> {
        let path = tensor(vec![2, 3, 2], DType::F32, vec![0; 2 * 3 * 2 * 4])?;
        let visibility = tensor(vec![2, 3], DType::Bool, vec![1; 2 * 3])?;
        let path_storage_id = path.storage_id();
        let visibility_storage_id = visibility.storage_id();
        let expected_storage = path.storage_byte_len() + visibility.storage_byte_len();
        let payload = NativeTracksPayload::checked(path, visibility)?;

        payload.validate()?;
        let parts = payload.resident_parts()?;
        assert_eq!(payload.track_path().descriptor().shape(), &[2, 3, 2]);
        assert_eq!(
            parts.owned_bytes(),
            u64::try_from(mem::size_of::<NativeTracksPayload>())?
        );
        let mut expected_storage_ids = vec![path_storage_id, visibility_storage_id];
        expected_storage_ids.sort_unstable_by_key(|storage_id| storage_id.get());
        assert_eq!(
            parts
                .tensor_allocations()
                .iter()
                .map(NativeMediaTensorResidentAllocation::storage_id)
                .collect::<Vec<_>>(),
            expected_storage_ids
        );
        assert_eq!(parts.resident_bytes()?, payload.resident_bytes());
        assert_eq!(
            payload.resident_bytes(),
            u64::try_from(mem::size_of::<NativeTracksPayload>())? + expected_storage
        );

        let wrong_path = tensor(vec![2, 3, 3], DType::F32, vec![0; 2 * 3 * 3 * 4])?;
        let visibility = tensor(vec![2, 3], DType::Bool, vec![1; 2 * 3])?;
        assert!(NativeTracksPayload::checked(wrong_path, visibility).is_err());
        Ok(())
    }

    #[test]
    fn sam3_track_data_accepts_empty_and_packed_source_variants() -> Result<(), Box<dyn Error>> {
        let empty = NativeSam3TrackDataPayload::checked(None, 3, vec![], 720, 1280)?;
        empty.validate()?;
        assert!(empty.packed_masks().is_none());
        assert!(empty.resident_parts()?.tensor_allocations().is_empty());
        assert_eq!(
            empty.resident_parts()?.owned_bytes(),
            empty.resident_bytes()
        );

        let packed = tensor(vec![3, 2, 8, 16], DType::U8, vec![0; 3 * 2 * 8 * 16])?;
        let packed_bytes = packed.storage_byte_len();
        let payload =
            NativeSam3TrackDataPayload::checked(Some(packed), 3, vec![1.0, 0.25], 720, 1280)?;
        payload.validate()?;
        let parts = payload.resident_parts()?;
        assert_eq!(payload.frame_count(), 3);
        assert_eq!(payload.scores(), &[1.0, 0.25]);
        assert_eq!(
            parts.owned_bytes(),
            u64::try_from(mem::size_of::<NativeSam3TrackDataPayload>())?
                + u64::try_from(2 * mem::size_of::<f64>())?
        );
        assert_eq!(parts.tensor_allocations().len(), 1);
        assert_eq!(
            parts
                .tensor_allocations()
                .first()
                .ok_or("SAM3 resident tensor allocation is absent")?
                .resident_bytes(),
            packed_bytes
        );
        assert_eq!(parts.resident_bytes()?, payload.resident_bytes());
        assert_eq!(
            payload.resident_bytes(),
            u64::try_from(mem::size_of::<NativeSam3TrackDataPayload>())?
                + u64::try_from(2 * mem::size_of::<f64>())?
                + packed_bytes
        );

        let wrong_objects = tensor(vec![3, 2, 8, 16], DType::U8, vec![0; 3 * 2 * 8 * 16])?;
        assert!(
            NativeSam3TrackDataPayload::checked(Some(wrong_objects), 3, vec![1.0], 720, 1280,)
                .is_err()
        );
        assert!(NativeSam3TrackDataPayload::checked(None, 3, vec![1.0], 720, 1280).is_err());
        Ok(())
    }

    #[test]
    fn resident_parts_deduplicate_aliases_and_preserve_distinct_storage_identity()
    -> Result<(), Box<dyn Error>> {
        let packed_bytes = vec![0; 3 * 2 * 8 * 16];
        let first_tensor = tensor(vec![3, 2, 8, 16], DType::U8, packed_bytes.clone())?;
        let second_tensor = tensor(vec![3, 2, 8, 16], DType::U8, packed_bytes)?;
        assert_ne!(first_tensor.storage_id(), second_tensor.storage_id());

        let aliased_parts = media_resident_parts(7, [&first_tensor, &first_tensor])?;
        assert_eq!(aliased_parts.tensor_allocations().len(), 1);
        assert_eq!(
            aliased_parts
                .tensor_allocations()
                .first()
                .ok_or("aliased resident tensor allocation is absent")?
                .storage_id(),
            first_tensor.storage_id()
        );

        let distinct_parts = media_resident_parts(7, [&first_tensor, &second_tensor])?;
        assert_eq!(distinct_parts.tensor_allocations().len(), 2);
        let mut distinct_allocations = distinct_parts.tensor_allocations().iter();
        let first_allocation = distinct_allocations
            .next()
            .ok_or("first distinct resident tensor allocation is absent")?;
        let second_allocation = distinct_allocations
            .next()
            .ok_or("second distinct resident tensor allocation is absent")?;
        assert!(first_allocation.storage_id().get() < second_allocation.storage_id().get());

        let first = Arc::new(NativeSam3TrackDataPayload::checked(
            Some(first_tensor),
            3,
            vec![1.0, 0.25],
            720,
            1280,
        )?);
        let outer_alias = first.clone();
        assert!(Arc::ptr_eq(&first, &outer_alias));
        assert_eq!(first.resident_parts()?, outer_alias.resident_parts()?);

        let second = NativeSam3TrackDataPayload::checked(
            Some(second_tensor),
            3,
            vec![1.0, 0.25],
            720,
            1280,
        )?;
        assert_eq!(
            first.semantic_digest_sha256(),
            second.semantic_digest_sha256()
        );
        let first_storage_id = first
            .resident_parts()?
            .tensor_allocations()
            .first()
            .ok_or("first SAM3 resident tensor allocation is absent")?
            .storage_id();
        let second_storage_id = second
            .resident_parts()?
            .tensor_allocations()
            .first()
            .ok_or("second SAM3 resident tensor allocation is absent")?
            .storage_id();
        assert_ne!(first_storage_id, second_storage_id);
        Ok(())
    }

    #[test]
    fn semantic_digests_are_allocation_independent() -> Result<(), Box<dyn Error>> {
        let construct = || -> Result<NativePoseKeypointPayload, NativeMediaPayloadError> {
            NativePoseKeypointPayload::checked(vec![NativePoseFrame::checked(
                512,
                512,
                vec![pose_person()?],
            )?])
        };
        let first = construct()?;
        let second = construct()?;
        assert_eq!(
            first.semantic_digest_sha256(),
            second.semantic_digest_sha256()
        );
        assert_eq!(first.resident_bytes(), second.resident_bytes());
        assert_eq!(NativeBoundingBoxPayload::SOURCE_TYPE_ID, "BOUNDING_BOX");
        assert_eq!(NativeFaceLandmarksPayload::SOURCE_TYPE_ID, "FACE_LANDMARKS");
        assert_eq!(NativePoseKeypointPayload::SOURCE_TYPE_ID, "POSE_KEYPOINT");
        assert_eq!(
            NativeSam3TrackDataPayload::SOURCE_TYPE_ID,
            "SAM3_TRACK_DATA"
        );
        assert_eq!(NativeTracksPayload::SOURCE_TYPE_ID, "TRACKS");
        Ok(())
    }

    #[test]
    fn tensor_semantic_digests_ignore_stream_and_bind_role_shape_and_content()
    -> Result<(), Box<dyn Error>> {
        let tracks = |stream,
                      path_shape: Vec<u64>,
                      visibility_shape: Vec<u64>,
                      path_bytes,
                      visibility_bytes| {
            NativeTracksPayload::checked(
                tensor_on_stream(path_shape, DType::F32, path_bytes, stream)?,
                tensor_on_stream(visibility_shape, DType::Bool, visibility_bytes, stream)?,
            )
            .map_err(Box::<dyn Error>::from)
        };
        let default_tracks = tracks(
            StreamId::DEFAULT,
            vec![2, 3, 2],
            vec![2, 3],
            vec![0; 2 * 3 * 2 * 4],
            vec![1; 2 * 3],
        )?;
        let other_stream_tracks = tracks(
            StreamId::new(17),
            vec![2, 3, 2],
            vec![2, 3],
            vec![0; 2 * 3 * 2 * 4],
            vec![1; 2 * 3],
        )?;
        assert_eq!(
            default_tracks.semantic_digest_sha256(),
            other_stream_tracks.semantic_digest_sha256()
        );

        let mut changed_path_bytes = vec![0; 2 * 3 * 2 * 4];
        *changed_path_bytes
            .first_mut()
            .ok_or("TRACKS content fixture must not be empty")? = 1;
        let changed_tracks = tracks(
            StreamId::DEFAULT,
            vec![2, 3, 2],
            vec![2, 3],
            changed_path_bytes,
            vec![1; 2 * 3],
        )?;
        assert_ne!(
            default_tracks.semantic_digest_sha256(),
            changed_tracks.semantic_digest_sha256()
        );
        let reshaped_tracks = tracks(
            StreamId::DEFAULT,
            vec![3, 2, 2],
            vec![3, 2],
            vec![0; 2 * 3 * 2 * 4],
            vec![1; 2 * 3],
        )?;
        assert_ne!(
            default_tracks.semantic_digest_sha256(),
            reshaped_tracks.semantic_digest_sha256()
        );

        let sam3 = |stream, shape: Vec<u64>, bytes| {
            NativeSam3TrackDataPayload::checked(
                Some(tensor_on_stream(shape, DType::U8, bytes, stream)?),
                3,
                vec![1.0, 0.25],
                720,
                1280,
            )
            .map_err(Box::<dyn Error>::from)
        };
        let default_sam3 = sam3(
            StreamId::DEFAULT,
            vec![3, 2, 8, 16],
            vec![0; 3 * 2 * 8 * 16],
        )?;
        let other_stream_sam3 = sam3(
            StreamId::new(17),
            vec![3, 2, 8, 16],
            vec![0; 3 * 2 * 8 * 16],
        )?;
        assert_eq!(
            default_sam3.semantic_digest_sha256(),
            other_stream_sam3.semantic_digest_sha256()
        );

        let mut changed_mask_bytes = vec![0; 3 * 2 * 8 * 16];
        *changed_mask_bytes
            .first_mut()
            .ok_or("SAM3 content fixture must not be empty")? = 1;
        let changed_sam3 = sam3(StreamId::DEFAULT, vec![3, 2, 8, 16], changed_mask_bytes)?;
        assert_ne!(
            default_sam3.semantic_digest_sha256(),
            changed_sam3.semantic_digest_sha256()
        );
        let reshaped_sam3 = sam3(
            StreamId::DEFAULT,
            vec![3, 2, 4, 32],
            vec![0; 3 * 2 * 8 * 16],
        )?;
        assert_ne!(
            default_sam3.semantic_digest_sha256(),
            reshaped_sam3.semantic_digest_sha256()
        );
        Ok(())
    }

    #[test]
    fn semantic_tensor_projection_binds_domain_role_and_dtype() -> Result<(), Box<dyn Error>> {
        fn digest(
            domain: &[u8],
            role: &[u8],
            tensor: &Tensor,
        ) -> Result<[u8; 32], NativeMediaPayloadError> {
            let mut projection = Projection::new::<NativeTracksPayload>(domain)?;
            projection.hash_tensor(role, tensor)?;
            Ok(projection.finish().0)
        }

        let f32_tensor = tensor(vec![1], DType::F32, vec![0; 4])?;
        let u32_tensor = tensor(vec![1], DType::U32, vec![0; 4])?;
        let baseline = digest(b"media-domain-a", b"tensor-role-a", &f32_tensor)?;
        assert_ne!(
            baseline,
            digest(b"media-domain-b", b"tensor-role-a", &f32_tensor)?
        );
        assert_ne!(
            baseline,
            digest(b"media-domain-a", b"tensor-role-b", &f32_tensor)?
        );
        assert_ne!(
            baseline,
            digest(b"media-domain-a", b"tensor-role-a", &u32_tensor)?
        );
        Ok(())
    }

    #[test]
    fn general_video_demux_decode_cancellable_projection_preserves_canonical_identity()
    -> Result<(), Box<dyn Error>> {
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let waveform = tensor(
            vec![1, 1, 4],
            DType::F32,
            f32_bytes(&[0.0, 0.25, -0.25, 1.0]),
        )?;
        let frames = tensor(vec![1, 2, 2, 3], DType::F32, f32_bytes(&[0.25; 12]))?;
        let cancellation = CancellationToken::default();
        let canonical_audio = NativeAudioPayload::checked(waveform.clone(), 48_000)?;
        let cancellable_audio =
            NativeAudioPayload::checked_cancellable(waveform, 48_000, &cancellation)?;
        assert_eq!(
            canonical_audio.semantic_digest_sha256(),
            cancellable_audio.semantic_digest_sha256()
        );
        assert_eq!(
            canonical_audio.resident_bytes(),
            cancellable_audio.resident_bytes()
        );
        let metadata = BTreeMap::from([("codec".to_owned(), "fixture".to_owned())]);
        let canonical_video = NativeVideoPayload::checked(
            frames.clone(),
            24,
            1,
            NativeVideoBitDepth::Eight,
            Some(canonical_audio),
            None,
            metadata.clone(),
        )?;
        let cancellable_video = NativeVideoPayload::checked_cancellable(
            frames.clone(),
            24,
            1,
            NativeVideoBitDepth::Eight,
            Some(cancellable_audio),
            None,
            metadata,
            &cancellation,
        )?;
        assert_eq!(
            canonical_video.semantic_digest_sha256(),
            cancellable_video.semantic_digest_sha256()
        );
        assert_eq!(
            canonical_video.resident_bytes(),
            cancellable_video.resident_bytes()
        );
        cancellable_video.validate_components_cancellable(&cancellation)?;

        cancellation.cancel();
        assert!(matches!(
            NativeVideoPayload::checked_cancellable(
                frames,
                24,
                1,
                NativeVideoBitDepth::Eight,
                None,
                None,
                BTreeMap::new(),
                &cancellation,
            ),
            Err(NativeCancellableMediaPayloadError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn canonical_audio_admits_zero_samples_without_padding() -> Result<(), Box<dyn Error>> {
        let empty = NativeAudioPayload::checked(tensor(vec![1, 1, 0], DType::F32, Vec::new())?, 1)?;
        assert_eq!(empty.waveform().descriptor().shape(), [1, 1, 0]);
        assert_eq!(empty.waveform().contiguous_bytes()?.len(), 0);
        assert_eq!(empty.sample_rate(), 1);
        empty.validate()?;
        assert_eq!(
            empty.resident_parts()?.resident_bytes()?,
            empty.resident_bytes()
        );

        let stereo =
            NativeAudioPayload::checked(tensor(vec![1, 2, 0], DType::F32, Vec::new())?, 192_000)?;
        let maximum_rate =
            NativeAudioPayload::checked(tensor(vec![1, 1, 0], DType::F32, Vec::new())?, 384_000)?;
        assert_ne!(
            empty.semantic_digest_sha256(),
            stereo.semantic_digest_sha256()
        );
        assert_ne!(
            empty.semantic_digest_sha256(),
            maximum_rate.semantic_digest_sha256()
        );
        assert_eq!(stereo.sample_rate(), 192_000);
        assert_eq!(maximum_rate.sample_rate(), 384_000);

        let video = NativeVideoPayload::checked(
            tensor(vec![1, 1, 1, 3], DType::U8, vec![0; 3])?,
            24,
            1,
            NativeVideoBitDepth::Eight,
            Some(empty),
            None,
            BTreeMap::new(),
        )?;
        assert_eq!(
            video
                .components()
                .and_then(NativeVideoComponentsPayload::audio)
                .map(|audio| audio.waveform().descriptor().shape()),
            Some([1, 1, 0].as_slice())
        );

        for (shape, bytes) in [
            (vec![0, 1, 0], Vec::new()),
            (vec![1, 0, 0], Vec::new()),
            (vec![1, 1], vec![0; 4]),
            (vec![1, 1, 0, 1], Vec::new()),
        ] {
            assert!(matches!(
                NativeAudioPayload::checked(tensor(shape, DType::F32, bytes)?, 48_000),
                Err(NativeMediaPayloadError::InvalidAudio)
            ));
        }
        for sample_rate in [0, 384_001] {
            assert!(matches!(
                NativeAudioPayload::checked(
                    tensor(vec![1, 1, 0], DType::F32, Vec::new())?,
                    sample_rate,
                ),
                Err(NativeMediaPayloadError::InvalidAudio)
            ));
        }
        let nonfinite = f32::NAN.to_ne_bytes().to_vec();
        assert!(matches!(
            NativeAudioPayload::checked(tensor(vec![1, 1, 1], DType::F32, nonfinite)?, 48_000),
            Err(NativeMediaPayloadError::InvalidAudio)
        ));

        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            NativeAudioPayload::checked_cancellable(
                tensor(vec![1, 1, 0], DType::F32, Vec::new())?,
                48_000,
                &cancellation,
            ),
            Err(NativeCancellableMediaPayloadError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn general_video_demux_decode_cancels_chunked_media_scans_and_rejects_projection_drift()
    -> Result<(), Box<dyn Error>> {
        let value_count = 3 * 64 * 1024 / size_of::<f32>();
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let values = vec![0.25_f32; value_count];
        let waveform = tensor(
            vec![1, 1, u64::try_from(value_count)?],
            DType::F32,
            f32_bytes(&values),
        )?;

        let finite_cancellation = CancellationToken::default();
        set_media_test_cancel_after_checks(Some(1));
        assert!(matches!(
            validate_finite_f32_cancellable(
                &waveform,
                NativeMediaPayloadError::InvalidAudio,
                &finite_cancellation,
            ),
            Err(NativeCancellableMediaPayloadError::Cancelled)
        ));

        let hash_cancellation = CancellationToken::default();
        let mut projection = Projection::new::<NativeAudioPayload>(b"test.chunked.audio")?;
        set_media_test_cancel_after_checks(Some(2));
        assert!(matches!(
            projection.hash_tensor_cancellable(b"waveform", &waveform, &hash_cancellation),
            Err(NativeCancellableMediaPayloadError::Cancelled)
        ));

        let frame_width = u64::try_from(value_count)?;
        let frames = tensor(
            vec![1, 1, frame_width, 3],
            DType::U8,
            vec![127; value_count * 3],
        )?;
        let alpha = tensor(vec![1, 1, frame_width, 1], DType::F32, f32_bytes(&values))?;
        let alpha_cancellation = CancellationToken::default();
        set_media_test_cancel_after_checks(Some(2));
        assert!(matches!(
            validate_video_cancellable(
                &frames,
                24,
                1,
                NativeVideoBitDepth::Eight,
                None,
                Some(&alpha),
                &BTreeMap::new(),
                &alpha_cancellation,
            ),
            Err(NativeCancellableMediaPayloadError::Cancelled)
        ));

        let cancellation = CancellationToken::default();
        let mut audio = NativeAudioPayload::checked_cancellable(waveform, 48_000, &cancellation)?;
        audio.semantic_digest_sha256[0] ^= 1;
        assert!(audio.validate_cancellable(&cancellation).is_err());

        let mut video = NativeVideoPayload::checked_cancellable(
            frames,
            24,
            1,
            NativeVideoBitDepth::Eight,
            None,
            Some(alpha),
            BTreeMap::new(),
            &cancellation,
        )?;
        let NativeVideoPayload::Components(components) = &mut video else {
            return Err("expected component video".into());
        };
        components.resident_bytes = components
            .resident_bytes
            .checked_add(1)
            .ok_or("resident byte test overflow")?;
        assert!(
            video
                .validate_components_cancellable(&cancellation)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn canonical_audio_video_and_artifact_payloads_are_bounded_and_placement_independent()
    -> Result<(), Box<dyn Error>> {
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let waveform_values = [0.0_f32, 0.25, -0.25, 1.0];
        let default_waveform = tensor_on_stream(
            vec![1, 1, 4],
            DType::F32,
            f32_bytes(&waveform_values),
            StreamId::DEFAULT,
        )?;
        let other_waveform = tensor_on_stream(
            vec![1, 1, 4],
            DType::F32,
            f32_bytes(&waveform_values),
            StreamId::new(91),
        )?;
        let audio = NativeAudioPayload::checked(default_waveform, 48_000)?;
        let other_audio = NativeAudioPayload::checked(other_waveform, 48_000)?;
        audio.validate()?;
        assert_eq!(
            audio.semantic_digest_sha256(),
            other_audio.semantic_digest_sha256()
        );
        assert_eq!(
            audio.resident_parts()?.resident_bytes()?,
            audio.resident_bytes()
        );
        assert!(
            NativeAudioPayload::checked(
                tensor(vec![1, 4], DType::F32, f32_bytes(&waveform_values))?,
                48_000,
            )
            .is_err()
        );

        let video = NativeVideoPayload::checked(
            tensor(vec![2, 2, 2, 3], DType::U8, vec![17; 24])?,
            30_000,
            1_001,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::from([("codec".to_owned(), "fixture".to_owned())]),
        )?;
        video.validate()?;
        assert_eq!(video.frame_rate(), (30_000, 1_001));
        assert_eq!(video.bit_depth(), NativeVideoBitDepth::Eight);
        assert_eq!(video.dimensions(), (2, 2));
        assert!((video.duration_seconds() - (2.0 * 1_001.0 / 30_000.0)).abs() < f64::EPSILON);
        assert_eq!(
            video.resident_parts()?.resident_bytes()?,
            video.resident_bytes()
        );
        let components = video
            .components()
            .ok_or(NativeMediaPayloadError::InvalidVideo)?;
        let alpha = tensor(
            vec![2, 2, 2, 1],
            DType::F32,
            f32_bytes(&[0.0, 0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0]),
        )?;
        let ten_bit = NativeVideoPayload::checked(
            components.frames().clone(),
            1_054_475_631_502_295,
            35_184_372_088_832,
            NativeVideoBitDepth::Ten,
            Some(audio),
            Some(alpha),
            components.metadata().clone(),
        )?;
        assert_eq!(ten_bit.bit_depth().bits(), 10);
        let ten_bit_components = ten_bit
            .components()
            .ok_or(NativeMediaPayloadError::InvalidVideo)?;
        assert!(ten_bit_components.audio().is_some());
        assert!(ten_bit_components.alpha().is_some());
        assert_ne!(
            video.semantic_digest_sha256(),
            ten_bit.semantic_digest_sha256()
        );
        assert!(NativeVideoBitDepth::try_from(9).is_err());
        assert!(
            NativeVideoPayload::checked(
                components.frames().clone(),
                60,
                2,
                NativeVideoBitDepth::Eight,
                None,
                None,
                BTreeMap::new(),
            )
            .is_err()
        );
        assert!(
            NativeVideoPayload::checked(
                components.frames().clone(),
                30_000,
                1_001,
                NativeVideoBitDepth::Eight,
                None,
                Some(tensor(vec![1, 2, 2, 1], DType::F32, vec![0; 16])?),
                BTreeMap::new(),
            )
            .is_err()
        );

        let svg = NativeArtifactPayload::checked(
            NativeArtifactKind::Svg,
            "image/svg+xml".to_owned(),
            b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>".to_vec(),
        )?;
        svg.validate()?;
        assert_eq!(svg.source_type_id(), "SVG");
        assert!(
            NativeArtifactPayload::checked(
                NativeArtifactKind::Svg,
                "Image/SVG".to_owned(),
                vec![1],
            )
            .is_err()
        );

        let ply = NativeFile3DPayload::checked(
            NativeFile3DRole::Ply,
            NativeFile3DFormat::Ply,
            b"ply\nformat ascii 1.0\nend_header\n".to_vec(),
        )?;
        ply.validate()?;
        assert_eq!(ply.source_type_id(), "FILE_3D_PLY");
        assert!(
            NativeFile3DPayload::checked(NativeFile3DRole::Spz, NativeFile3DFormat::Ply, vec![1],)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn encoded_h264_mp4_video_binds_component_identity_and_portable_storage()
    -> Result<(), Box<dyn Error>> {
        let source = NativeVideoPayload::checked(
            tensor(
                vec![1, 2, 2, 3],
                DType::F32,
                [0.0_f32; 12]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect(),
            )?,
            30,
            1,
            NativeVideoBitDepth::Eight,
            None,
            None,
            BTreeMap::from([("prompt".to_owned(), "source-only".to_owned())]),
        )?;
        let content = b"HMP4";
        let bytes = tensor(vec![content.len() as u64], DType::U8, content.to_vec())?;
        let storage_id = bytes.storage_id();
        let content_sha256 = Sha256::digest(content).into();
        let encoded = NativeVideoPayload::checked_h264_mp4_from_component(
            &source,
            bytes,
            content_sha256,
            (2, 2),
            (30, 1),
            1,
        )?;
        encoded.validate()?;
        assert_eq!(encoded.representation(), NativeVideoRepresentation::Encoded);
        assert!(encoded.components().is_none());
        let backing = encoded
            .encoded()
            .ok_or(NativeMediaPayloadError::InvalidVideo)?;
        assert_eq!(backing.bytes().storage_id(), storage_id);
        assert_eq!(backing.content_sha256(), &content_sha256);
        assert_eq!(
            backing.source_video_sha256(),
            source.semantic_digest_sha256()
        );
        assert_eq!(backing.container(), crate::NativeVideoContainer::Mp4);
        assert_eq!(backing.codec(), crate::NativeVideoCodec::H264);
        assert_eq!(
            backing.pixel_format(),
            crate::NativeVideoPixelFormat::Yuv420p
        );
        assert_eq!(backing.bit_depth(), NativeVideoBitDepth::Eight);
        assert!(!backing.has_audio());
        assert!(!backing.has_alpha());
        assert_eq!(encoded.resident_parts()?.tensor_allocations().len(), 1);
        assert_eq!(
            encoded.resident_parts()?.resident_bytes()?,
            encoded.resident_bytes()
        );
        assert_ne!(
            encoded.semantic_digest_sha256(),
            source.semantic_digest_sha256()
        );
        assert_eq!(
            encoded.semantic_digest_sha256(),
            &[
                111, 135, 190, 43, 48, 85, 92, 98, 153, 27, 148, 91, 170, 3, 175, 155, 247, 61,
                203, 233, 6, 193, 133, 66, 59, 139, 38, 217, 163, 124, 195, 26,
            ]
        );

        let ten_bit_source = NativeVideoPayload::checked(
            source
                .components()
                .ok_or(NativeMediaPayloadError::InvalidVideo)?
                .frames()
                .clone(),
            30,
            1,
            NativeVideoBitDepth::Ten,
            None,
            None,
            BTreeMap::from([("prompt".to_owned(), "source-only".to_owned())]),
        )?;
        let ten_bit_content = b"H10P4";
        let ten_bit_bytes = tensor(
            vec![ten_bit_content.len() as u64],
            DType::U8,
            ten_bit_content.to_vec(),
        )?;
        let ten_bit_storage_id = ten_bit_bytes.storage_id();
        let ten_bit_content_sha256 = Sha256::digest(ten_bit_content).into();
        let ten_bit = NativeVideoPayload::checked_h264_mp4_from_component(
            &ten_bit_source,
            ten_bit_bytes,
            ten_bit_content_sha256,
            (2, 2),
            (30, 1),
            1,
        )?;
        ten_bit.validate()?;
        let ten_bit_backing = ten_bit
            .encoded()
            .ok_or(NativeMediaPayloadError::InvalidVideo)?;
        assert_eq!(ten_bit_backing.bytes().storage_id(), ten_bit_storage_id);
        assert_eq!(ten_bit_backing.bit_depth(), NativeVideoBitDepth::Ten);
        assert_eq!(
            ten_bit_backing.pixel_format(),
            crate::NativeVideoPixelFormat::Yuv420p10le
        );
        assert_eq!(
            ten_bit_backing.source_video_sha256(),
            ten_bit_source.semantic_digest_sha256()
        );
        assert_ne!(
            ten_bit.semantic_digest_sha256(),
            encoded.semantic_digest_sha256()
        );
        assert_eq!(ten_bit.resident_parts()?.tensor_allocations().len(), 1);

        let forged = tensor(vec![content.len() as u64], DType::U8, content.to_vec())?;
        assert!(
            NativeVideoPayload::checked_h264_mp4_from_component(
                &source,
                forged,
                [0; 32],
                (2, 2),
                (30, 1),
                1,
            )
            .is_err()
        );
        let mismatched = tensor(vec![content.len() as u64], DType::U8, content.to_vec())?;
        assert!(
            NativeVideoPayload::checked_h264_mp4_from_component(
                &source,
                mismatched,
                content_sha256,
                (3, 2),
                (30, 1),
                1,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn canonical_splat_mesh_voxel_and_camera_payloads_bind_topology_and_residency()
    -> Result<(), Box<dyn Error>> {
        let f32_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let positions = tensor(
            vec![1, 2, 3],
            DType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
        )?;
        let scales = tensor(vec![1, 2, 3], DType::F32, f32_bytes(&[0.1; 6]))?;
        let rotations = tensor(
            vec![1, 2, 4],
            DType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        )?;
        let opacity = tensor(vec![1, 2, 1], DType::F32, f32_bytes(&[0.0; 2]))?;
        let spherical_harmonics = tensor(vec![1, 2, 1, 3], DType::F32, f32_bytes(&[0.5; 6]))?;
        let uniform_counts = NativeSplatPayload::checked(
            positions.clone(),
            scales.clone(),
            rotations.clone(),
            opacity.clone(),
            spherical_harmonics.clone(),
            Some(vec![2]),
        )?;
        let splat = NativeSplatPayload::checked(
            positions,
            scales,
            rotations,
            opacity,
            spherical_harmonics,
            None,
        )?;
        splat.validate()?;
        assert!(uniform_counts.counts().is_none());
        assert_eq!(
            uniform_counts.semantic_digest_sha256(),
            splat.semantic_digest_sha256()
        );
        assert_eq!(uniform_counts.resident_bytes(), splat.resident_bytes());
        assert_eq!((splat.batch_count(), splat.splat_count()), (1, 2));
        assert_eq!(
            splat.resident_parts()?.resident_bytes()?,
            splat.resident_bytes()
        );

        let padded_splat = NativeSplatPayload::checked(
            tensor(
                vec![1, 2, 3],
                DType::F32,
                f32_bytes(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )?,
            tensor(
                vec![1, 2, 3],
                DType::F32,
                f32_bytes(&[0.1, 0.1, 0.1, 0.0, 0.0, 0.0]),
            )?,
            tensor(
                vec![1, 2, 4],
                DType::F32,
                f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            )?,
            tensor(vec![1, 2, 1], DType::F32, f32_bytes(&[0.5, 0.0]))?,
            tensor(
                vec![1, 2, 1, 3],
                DType::F32,
                f32_bytes(&[0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
            )?,
            Some(vec![1]),
        )?;
        assert_eq!(padded_splat.counts(), Some(&[1][..]));
        assert_eq!(padded_splat.splat_count(), 1);
        assert!(matches!(
            NativeSplatPayload::checked(
                tensor(
                    vec![1, 2, 3],
                    DType::F32,
                    f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                )?,
                padded_splat.scales().clone(),
                padded_splat.rotations().clone(),
                padded_splat.opacity().clone(),
                padded_splat.spherical_harmonics().clone(),
                Some(vec![1]),
            ),
            Err(NativeMediaPayloadError::InvalidSplat)
        ));

        let vertices = tensor(
            vec![3, 3],
            DType::F32,
            f32_bytes(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )?;
        let faces = tensor(
            vec![1, 3],
            DType::I32,
            [0_i32, 1, 2]
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect(),
        )?;
        let full_mesh = NativeMeshPayload::checked(
            vec![NativeMeshBatch::checked(
                vertices.clone(),
                faces.clone(),
                Some(vertices.clone()),
                Some(tensor(
                    vec![3, 2],
                    DType::F32,
                    f32_bytes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
                )?),
                Some(tensor(vec![3, 4], DType::F32, f32_bytes(&[1.0; 12]))?),
                Some(tensor(vec![1, 1, 3], DType::F32, f32_bytes(&[0.25; 3]))?),
            )?],
            false,
        )?;
        full_mesh.validate()?;
        assert!(!full_mesh.unlit());
        assert!(full_mesh.batches()[0].normals().is_some());
        assert!(full_mesh.batches()[0].uvs().is_some());
        assert!(full_mesh.batches()[0].colors().is_some());
        assert!(full_mesh.batches()[0].texture().is_some());
        assert_eq!(
            full_mesh.resident_parts()?.tensor_allocations().len(),
            5,
            "vertices and aliased normals must charge one storage"
        );
        assert!(matches!(
            NativeMeshBatch::checked(
                vertices.clone(),
                tensor(
                    vec![1, 3],
                    DType::I32,
                    [0_i32, 1, 3]
                        .into_iter()
                        .flat_map(i32::to_ne_bytes)
                        .collect(),
                )?,
                None,
                None,
                None,
                None,
            ),
            Err(NativeMediaPayloadError::InvalidMesh)
        ));
        let mesh = NativeMeshPayload::checked(
            vec![NativeMeshBatch::checked(
                vertices, faces, None, None, None, None,
            )?],
            true,
        )?;
        mesh.validate()?;
        assert!(mesh.unlit());
        assert_eq!(mesh.batches().len(), 1);
        assert_eq!(
            mesh.resident_parts()?.resident_bytes()?,
            mesh.resident_bytes()
        );

        let density = tensor(vec![1, 2, 2, 2], DType::F32, f32_bytes(&[0.0; 8]))?;
        let colors = tensor(vec![1, 2, 2, 2, 3], DType::F32, f32_bytes(&[0.25; 24]))?;
        let voxel = NativeVoxelPayload::checked(density, Some(colors), identity_matrix())?;
        voxel.validate()?;
        assert_eq!(
            voxel.resident_parts()?.resident_bytes()?,
            voxel.resident_bytes()
        );

        let camera = NativeCameraPayload::checked(
            NativeCameraRole::Load3D,
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            1.0,
            None,
            NativeCameraProjection::Perspective {
                fov_degrees: 45.0,
                aspect_ratio: 4.0 / 3.0,
                near: 0.01,
                far: 100.0,
            },
            1_024,
            768,
        )?;
        camera.validate()?;
        assert_eq!(camera.source_type_id(), "LOAD3D_CAMERA");
        assert!(
            NativeCameraPayload::checked(
                NativeCameraRole::Load3D,
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                1.0,
                None,
                NativeCameraProjection::Perspective {
                    fov_degrees: 45.0,
                    aspect_ratio: 4.0 / 3.0,
                    near: 0.01,
                    far: 100.0,
                },
                0,
                768,
            )
            .is_err()
        );
        Ok(())
    }

    fn identity_matrix() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }
}
