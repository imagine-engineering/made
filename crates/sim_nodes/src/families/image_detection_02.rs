use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativePrimitiveType, NativeSdPoseRequest, NativeStoredPayload,
    NativeTypeUnion, NativeValue, NativeValueType, built_in_source_schema,
};
use sim_media::{NativeBoundingBox, NativeBoundingBoxPayload};
use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["SDPoseFaceBBoxes", "SDPoseKeypointExtractor"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

#[derive(Clone, Copy, Debug)]
enum SdPoseNode {
    FaceBoxes,
    Extract,
}

fn handle_type(name: &str) -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(
        match name {
            "MODEL" => NativeHandleKind::Model,
            "VAE" => NativeHandleKind::Vae,
            "IMAGE" => NativeHandleKind::Image,
            _ => NativeHandleKind::StructuredCompute,
        },
        name,
    )
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [SdPoseNode::FaceBoxes, SdPoseNode::Extract].into_iter().map(|node| {
        let (feature_id, ports, output, display_name, description, aliases) = match node {
            SdPoseNode::FaceBoxes => (
                "COMFY-NODE-0607", vec![("keypoints", "POSE_KEYPOINT", true), ("scale", "FLOAT", true), ("force_square", "BOOLEAN", true)],
                ("bboxes", "BOUNDING_BOX"), "SDPose Face Bounding Boxes", "",
                vec!["face bbox", "face bounding box", "pose", "keypoints"],
            ),
            SdPoseNode::Extract => (
                "COMFY-NODE-0608", vec![("model", "MODEL", true), ("vae", "VAE", true), ("image", "IMAGE", true), ("batch_size", "INT", true), ("bboxes", "BOUNDING_BOX", false)],
                ("keypoints", "POSE_KEYPOINT"), "SDPose Keypoint Extractor",
                "Extract pose keypoints from images using the SDPose model: https://huggingface.co/Comfy-Org/SDPose/tree/main/checkpoints",
                vec!["openpose", "pose detection", "preprocessor", "keypoints", "sdpose"],
            ),
        };
        let names = ports.iter().map(|(name, _, _)| (*name).to_owned()).collect::<Vec<_>>();
        let schema = built_in_source_schema(node.class_type())
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
            .bind_execution_ports(&names, &[], &[output.0.to_owned()])
            .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
        let inputs = ports.into_iter().map(|(name, kind, required)| {
            let primitive = match kind { "FLOAT" => Some(NativePrimitiveType::Number), "INT" => Some(NativePrimitiveType::Integer), "BOOLEAN" => Some(NativePrimitiveType::Boolean), _ => None };
            Ok(NativeInputDescriptor {
                name: name.to_owned(), accepted_types: NativeTypeUnion::new([match primitive {
                    Some(value) => NativeValueType::Primitive(value), None => NativeValueType::Handle(handle_type(kind)?),
                }])?, required, hidden: false, lazy: false, cardinality: NativePortCardinality::Scalar, allows_literal: primitive.is_some(),
            })
        }).collect::<Result<Vec<_>, NativeNodeContractError>>()?;
        Ok(NativeNodeBinding::Executable {
            feature_id: feature_id.to_owned(),
            descriptor: NativeNodeDescriptor {
                schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION, class_type: node.class_type().to_owned(), implementation_version: node.implementation_version().to_owned(),
                source_schema: Some(schema), inputs, dynamic_inputs: Vec::new(),
                outputs: vec![NativeOutputDescriptor { name: output.0.to_owned(), produced_type: NativeValueType::Handle(handle_type(output.1)?), is_list: false }],
                output_node: false, effect: NativeEffectClass::Pure, cache: NativeCachePolicy::InputIdentity,
            },
            presentation: NativeNodePresentation { display_name: display_name.to_owned(), category: "image/detection".to_owned(), description: description.to_owned(), output_names: vec![output.0.to_owned()], search_aliases: aliases.into_iter().map(str::to_owned).collect(), is_deprecated: false, is_experimental: false },
            node: Arc::new(node),
        })
    }).collect()
}

impl SdPoseNode {
    fn validate_inputs(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(), NativeNodeFailure> {
        match self {
            Self::FaceBoxes => {
                if inputs.len() != 3 {
                    return Err(invalid("Face bounding boxes require three scalar inputs"));
                }
                input_handle(inputs, "keypoints", "POSE_KEYPOINT")?;
                face_parameters(inputs)?;
            }
            Self::Extract => {
                if inputs.len() != 4 + usize::from(inputs.contains_key("bboxes")) {
                    return Err(invalid("Invalid SDPose extraction inputs"));
                }
                input_handle(inputs, "model", "MODEL")?;
                input_handle(inputs, "vae", "VAE")?;
                input_handle(inputs, "image", "IMAGE")?;
                batch_size(inputs)?;
                if inputs.contains_key("bboxes") {
                    input_handle(inputs, "bboxes", "BOUNDING_BOX")?;
                }
            }
        }
        Ok(())
    }
}

impl NativeNode for SdPoseNode {
    fn class_type(&self) -> &str {
        match self {
            Self::FaceBoxes => "SDPoseFaceBBoxes",
            Self::Extract => "SDPoseKeypointExtractor",
        }
    }
    fn implementation_version(&self) -> &str {
        "source-d9b38524-sdpose-v1"
    }
    fn demanded_lazy_inputs(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<BTreeSet<String>, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        Ok(BTreeSet::new())
    }
    fn cache_change_token(
        &self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<String, NativeNodeFailure> {
        self.validate_inputs(inputs)?;
        Ok(self.implementation_version().to_owned())
    }
    fn cache_dependencies(
        &self,
        context: &NativeNodeContext,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<NativeCacheDependencies, NativeNodeFailure> {
        check(context)?;
        self.validate_inputs(inputs)?;
        let mut artifact_digests = BTreeMap::new();
        for (name, value) in inputs {
            if let NativeValue::Handle { value: handle } = value {
                let payload = context
                    .handle_store()
                    .resolve(handle, handle.handle_type(), &context.cancellation)
                    .map_err(|error| failure(context, error))?;
                artifact_digests.insert(name.clone(), payload.digest_sha256());
            }
        }
        Ok(NativeCacheDependencies {
            artifact_digests,
            ..Default::default()
        })
    }
    fn execute<'a>(
        &'a self,
        context: NativeNodeContext,
        inputs: BTreeMap<String, NativeValue>,
    ) -> BoxFuture<'a, Result<NativeNodeOutcome, NativeNodeFailure>> {
        Box::pin(async move {
            check(&context)?;
            self.validate_inputs(&inputs)?;
            let payload = match self {
                Self::FaceBoxes => {
                    let handle = input_handle(&inputs, "keypoints", "POSE_KEYPOINT")?;
                    let resolved = context
                        .handle_store()
                        .resolve(
                            handle,
                            &handle_type("POSE_KEYPOINT").map_err(invalid)?,
                            &context.cancellation,
                        )
                        .map_err(|error| failure(&context, error))?;
                    let NativeStoredPayload::PoseKeypoint(pose) = resolved.as_ref() else {
                        return Err(invalid("Expected canonical pose keypoints"));
                    };
                    let (scale, force_square) = face_parameters(&inputs)?;
                    let mut frames = Vec::new();
                    for frame in pose.frames() {
                        check(&context)?;
                        let mut boxes = Vec::new();
                        for person in frame.people() {
                            check(&context)?;
                            if let Some([x, y, width, height]) = source_face_box(
                                frame.canvas_width(),
                                frame.canvas_height(),
                                person.face().iter().map(|point| (point.x(), point.y())),
                                scale,
                                force_square,
                            )
                            .map_err(invalid)?
                            {
                                boxes.push(
                                    NativeBoundingBox::checked(
                                        f64::from(x),
                                        f64::from(y),
                                        f64::from(width),
                                        f64::from(height),
                                        None,
                                        None,
                                    )
                                    .map_err(invalid)?,
                                );
                            }
                        }
                        frames.push(boxes);
                    }
                    NativeStoredPayload::BoundingBox(Arc::new(
                        NativeBoundingBoxPayload::checked(frames).map_err(invalid)?,
                    ))
                }
                Self::Extract => {
                    let request = NativeSdPoseRequest {
                        model: input_handle(&inputs, "model", "MODEL")?.clone(),
                        vae: input_handle(&inputs, "vae", "VAE")?.clone(),
                        image: input_handle(&inputs, "image", "IMAGE")?.clone(),
                        batch_size: batch_size(&inputs)?,
                        bounding_boxes: inputs
                            .get("bboxes")
                            .map(|_| input_handle(&inputs, "bboxes", "BOUNDING_BOX").cloned())
                            .transpose()?,
                    };
                    NativeStoredPayload::PoseKeypoint(Arc::new(
                        context.extract_native_sdpose(&request)?,
                    ))
                }
            };
            check(&context)?;
            let handle = context
                .handle_store()
                .publish(payload, &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let outcome = NativeNodeOutcome::Values {
                outputs: vec![NativeValue::Handle {
                    value: handle.clone(),
                }],
                ui: None,
                effects: Vec::new(),
            };
            if let Err(error) = check(&context).and_then(|()| outcome.validate().map_err(invalid)) {
                context
                    .handle_store()
                    .revoke(&handle, &sim_types::CancellationToken::default())
                    .map_err(|error| failure(&context, error))?;
                return Err(error);
            }
            Ok(outcome)
        })
    }
}

fn input_handle<'a>(
    inputs: &'a BTreeMap<String, NativeValue>,
    name: &str,
    kind: &str,
) -> Result<&'a NativeOpaqueHandle, NativeNodeFailure> {
    match inputs.get(name) {
        Some(NativeValue::Handle { value })
            if value.handle_type() == &handle_type(kind).map_err(invalid)? =>
        {
            Ok(value)
        }
        _ => Err(invalid(format!("{name} must be a scalar {kind} handle"))),
    }
}
fn batch_size(inputs: &BTreeMap<String, NativeValue>) -> Result<usize, NativeNodeFailure> {
    let value = match inputs.get("batch_size") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => usize::try_from(*value).ok(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => usize::try_from(*value).ok(),
        _ => None,
    };
    value
        .filter(|value| (1..=10_000).contains(value))
        .ok_or_else(|| invalid("batch_size must be an integer between 1 and 10000"))
}
fn face_parameters(
    inputs: &BTreeMap<String, NativeValue>,
) -> Result<(f64, bool), NativeNodeFailure> {
    let scale = match inputs.get("scale") {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Number(value),
        }) => *value,
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => *value as f64,
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => *value as f64,
        _ => return Err(invalid("scale must be a scalar number")),
    };
    if !scale.is_finite() || !(1.0..=10.0).contains(&scale) {
        return Err(invalid("scale must be between 1 and 10"));
    }
    let Some(NativeValue::Primitive {
        value: NativePrimitive::Boolean(force_square),
    }) = inputs.get("force_square")
    else {
        return Err(invalid("force_square must be a scalar boolean"));
    };
    Ok((scale, *force_square))
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_sdpose_inputs".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: "SDPose cancelled".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        invalid(error)
    }
}

fn source_face_box(
    width: u32,
    height: u32,
    points: impl Iterator<Item = (f64, f64)>,
    scale: f64,
    force_square: bool,
) -> Result<Option<[u32; 4]>, &'static str> {
    if width == 0 || height == 0 || !scale.is_finite() || !(1.0..=10.0).contains(&scale) {
        return Err("invalid face bounding-box parameters");
    }
    let mut minimum_x = f64::INFINITY;
    let mut minimum_y = f64::INFINITY;
    let mut maximum_x = f64::NEG_INFINITY;
    let mut maximum_y = f64::NEG_INFINITY;
    for (x, y) in points {
        // NumPy normalizes float32 coordinates before promoting the pixel projection to float64.
        let x = (x as f32 / width as f32) as f64 * f64::from(width);
        let y = (y as f32 / height as f32) as f64 * f64::from(height);
        if !x.is_finite() || !y.is_finite() {
            return Err("face coordinates exceed source float32 range");
        }
        minimum_x = minimum_x.min(x);
        minimum_y = minimum_y.min(y);
        maximum_x = maximum_x.max(x);
        maximum_y = maximum_y.max(y);
    }
    let initial_width = maximum_x - minimum_x;
    let initial_height = maximum_y - minimum_y;
    if initial_width <= 0.0 || initial_height <= 0.0 {
        return Ok(None);
    }
    let expanded_area = initial_width * initial_height * scale;
    let new_width = (expanded_area * (initial_width / initial_height)).sqrt();
    let new_height = (expanded_area * (initial_height / initial_width)).sqrt();
    let horizontal_padding = (new_width - initial_width) / 2.0;
    let vertical_padding = (new_height - initial_height) / 4.0;
    let left = (minimum_x - horizontal_padding).max(0.0).trunc();
    let right = (maximum_x + horizontal_padding)
        .min(f64::from(width))
        .trunc();
    let top = (minimum_y - 3.0 * vertical_padding).max(0.0).trunc();
    let bottom = (maximum_y + vertical_padding)
        .min(f64::from(height))
        .trunc();
    if right <= left || bottom <= top {
        return Ok(None);
    }
    let (mut left, mut right, mut top, mut bottom) =
        (left as u64, right as u64, top as u64, bottom as u64);
    if force_square && right - left != bottom - top {
        let side = (right - left).max(bottom - top);
        let center_x = (left + right) / 2;
        let center_y = (top + bottom) / 2;
        let half = side / 2;
        left = center_x.saturating_sub(half);
        top = center_y.saturating_sub(half);
        right = u64::from(width).min(left + side);
        bottom = u64::from(height).min(top + side);
        left = right.saturating_sub(side);
        top = bottom.saturating_sub(side);
    }
    Ok(Some([
        left as u32,
        top as u32,
        (right - left) as u32,
        (bottom - top) as u32,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::error::Error;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/image-detection-comfy-node-0607/fixture.json"
    ));

    #[test]
    fn sdpose_rows_preserve_exact_source_schema() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(FIXTURE)?;
        let rows = fixture["nodes"].as_array().ok_or("missing rows")?;
        let registry = crate::NodeRegistry::built_in()?;
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 2);
        assert_eq!(rows.len(), 2);
        for (binding, row) in bindings.iter().zip(rows) {
            registry.validate_native_binding(binding)?;
            let NativeNodeBinding::Executable {
                descriptor,
                feature_id,
                node,
                ..
            } = binding
            else {
                return Err("non-executable row".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(
                feature_id,
                row["feature_id"].as_str().ok_or("missing feature")?
            );
            assert_eq!(
                serde_json::to_value(built_in_source_schema(node.class_type())?)?,
                row["schema"]
            );
            assert_eq!(
                descriptor,
                &serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
            assert!(descriptor.inputs.iter().all(|input| !input.lazy
                && !input.hidden
                && input.cardinality == NativePortCardinality::Scalar));
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
        }
        Ok(())
    }

    #[derive(Deserialize)]
    struct Reference {
        cases: Vec<Case>,
        source_sha256: String,
    }
    #[derive(Deserialize)]
    struct Case {
        id: String,
        frames: Vec<Frame>,
        scale: f64,
        force_square: bool,
        expected: serde_json::Value,
    }
    #[derive(Deserialize)]
    struct Frame {
        canvas_width: u32,
        canvas_height: u32,
        people: Vec<Person>,
    }
    #[derive(Deserialize)]
    struct Person {
        face_keypoints_2d: Vec<f64>,
    }

    #[test]
    fn sdpose_face_geometry_matches_pinned_source_including_float32_rounding()
    -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(FIXTURE)?;
        let reference: Reference = serde_json::from_value(fixture["face_reference"].clone())?;
        assert_eq!(
            reference.source_sha256,
            "d9b38524b1ee4b09b0ba4373537fb53526f08a6b2d8f714bb80225379342ee21"
        );
        assert_eq!(reference.cases.len(), 48);
        for case in reference.cases {
            let mut frames = Vec::new();
            for frame in case.frames {
                let mut boxes = Vec::new();
                for person in frame.people {
                    assert_eq!(person.face_keypoints_2d.len(), 210);
                    let points = person
                        .face_keypoints_2d
                        .chunks_exact(3)
                        .map(|point| (point[0], point[1]));
                    if let Some([x, y, width, height]) = source_face_box(
                        frame.canvas_width,
                        frame.canvas_height,
                        points,
                        case.scale,
                        case.force_square,
                    )? {
                        boxes.push(serde_json::json!({"x":x,"y":y,"width":width,"height":height}));
                    }
                }
                frames.push(boxes);
            }
            assert_eq!(serde_json::to_value(frames)?, case.expected, "{}", case.id);
        }
        assert!(source_face_box(10, 10, [(f64::MAX, 0.0)].into_iter(), 1.5, false).is_err());
        assert!(source_face_box(10, 10, [(0.0, 0.0)].into_iter(), f64::NAN, false).is_err());
        Ok(())
    }
}
