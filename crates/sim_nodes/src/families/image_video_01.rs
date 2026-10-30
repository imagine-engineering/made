use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCacheDependencies, NativeCachePolicy,
    NativeEffectClass, NativeHandleKind, NativeHandleType, NativeInputDescriptor, NativeNode,
    NativeNodeBinding, NativeNodeBindingsFactory, NativeNodeContext, NativeNodeContractError,
    NativeNodeDescriptor, NativeNodeFailure, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodePresentation, NativeOpaqueHandle, NativeOutputDescriptor, NativePortCardinality,
    NativePrimitive, NativePrimitiveType, NativeStoredPayload, NativeTypeUnion, NativeValue,
    NativeValueType, built_in_source_schema,
};
use sim_media::NativeAudioPayload;
use sim_tensor::MemoryFormatReference;
use sim_tensor::generated_indexing_masking_01::narrow_function_exact_native;
use sim_tensor::generated_storage_dtype_device_01::contiguous_with_context_exact_native;
use sim_tensor::{
    CpuBackend, ExecutionContext, ImageTensor, NativeTensorPayload, NativeTensorRole,
};

use futures::future::BoxFuture;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &["WanDancerPadKeyframes", "WanDancerPadKeyframesList"];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;
const OUTPUT_NAMES: [&str; 3] = ["keyframes_sequence", "keyframes_mask", "audio_segment"];

#[derive(Clone, Copy, Debug)]
struct WanDancerPadding {
    list: bool,
}

fn handle_type(name: &str) -> Result<NativeHandleType, NativeNodeContractError> {
    NativeHandleType::new(
        match name {
            "IMAGE" => NativeHandleKind::Image,
            "MASK" => NativeHandleKind::Mask,
            _ => NativeHandleKind::Audio,
        },
        name,
    )
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    [false, true]
        .into_iter()
        .map(|list| {
            let node = WanDancerPadding { list };
            let names = ["images", "segment_length", node.selector(), "audio"].map(str::to_owned);
            let outputs = OUTPUT_NAMES.map(str::to_owned);
            let schema = built_in_source_schema(node.class_type())
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?
                .bind_execution_ports(&names, &[], &outputs)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let mut inputs = Vec::new();
            for (name, kind) in names.into_iter().zip(["IMAGE", "INT", "INT", "AUDIO"]) {
                inputs.push(NativeInputDescriptor {
                    name,
                    accepted_types: NativeTypeUnion::new([if kind == "INT" {
                        NativeValueType::Primitive(NativePrimitiveType::Integer)
                    } else {
                        NativeValueType::Handle(handle_type(kind)?)
                    }])?,
                    required: true,
                    hidden: false,
                    lazy: false,
                    cardinality: NativePortCardinality::Scalar,
                    allows_literal: kind == "INT",
                });
            }
            let outputs = OUTPUT_NAMES
                .into_iter()
                .zip(["IMAGE", "MASK", "AUDIO"])
                .map(|(name, kind)| {
                    Ok(NativeOutputDescriptor {
                        name: name.to_owned(),
                        produced_type: NativeValueType::Handle(handle_type(kind)?),
                        is_list: list,
                    })
                })
                .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
            Ok(NativeNodeBinding::Executable {
                feature_id: if list {
                    "COMFY-NODE-0763"
                } else {
                    "COMFY-NODE-0762"
                }
                .to_owned(),
                descriptor: NativeNodeDescriptor {
                    schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
                    class_type: node.class_type().to_owned(),
                    implementation_version: node.implementation_version().to_owned(),
                    source_schema: Some(schema),
                    inputs,
                    dynamic_inputs: Vec::new(),
                    outputs,
                    output_node: false,
                    effect: NativeEffectClass::Pure,
                    cache: NativeCachePolicy::InputIdentity,
                },
                presentation: NativeNodePresentation {
                    display_name: node.class_type().to_owned(),
                    category: "image/video".to_owned(),
                    description: String::new(),
                    output_names: OUTPUT_NAMES.map(str::to_owned).to_vec(),
                    search_aliases: Vec::new(),
                    is_deprecated: false,
                    is_experimental: false,
                },
                node: Arc::new(node),
            })
        })
        .collect()
}

impl WanDancerPadding {
    fn selector(self) -> &'static str {
        if self.list {
            "num_segments"
        } else {
            "segment_index"
        }
    }
    fn validate_inputs(
        self,
        inputs: &BTreeMap<String, NativeValue>,
    ) -> Result<(u64, u64), NativeNodeFailure> {
        if inputs.len() != 4 {
            return Err(invalid("Padding requires four scalar inputs"));
        }
        input_handle(inputs, "images", "IMAGE")?;
        input_handle(inputs, "audio", "AUDIO")?;
        Ok((
            integer(inputs, "segment_length", 1, 10000)?,
            integer(inputs, self.selector(), u64::from(self.list), 100)?,
        ))
    }
}

impl NativeNode for WanDancerPadding {
    fn class_type(&self) -> &str {
        if self.list {
            "WanDancerPadKeyframesList"
        } else {
            "WanDancerPadKeyframes"
        }
    }
    fn implementation_version(&self) -> &str {
        "source-5375a5c5-padding-v1"
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
        for (name, kind) in [("images", "IMAGE"), ("audio", "AUDIO")] {
            let handle = input_handle(inputs, name, kind)?;
            let payload = context
                .handle_store()
                .resolve(handle, handle.handle_type(), &context.cancellation)
                .map_err(|error| failure(context, error))?;
            artifact_digests.insert(name.to_owned(), payload.digest_sha256());
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
            let (length, selector) = self.validate_inputs(&inputs)?;
            let images = input_handle(&inputs, "images", "IMAGE")?;
            let audio = input_handle(&inputs, "audio", "AUDIO")?;
            let images = context
                .handle_store()
                .resolve(images, images.handle_type(), &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let audio = context
                .handle_store()
                .resolve(audio, audio.handle_type(), &context.cancellation)
                .map_err(|error| failure(&context, error))?;
            let NativeStoredPayload::Tensor(images) = images.as_ref() else {
                return Err(invalid("Expected canonical IMAGE payload"));
            };
            let images = images
                .image()
                .ok_or_else(|| invalid("Expected canonical ImageTensor"))?;
            let NativeStoredPayload::Audio(audio) = audio.as_ref() else {
                return Err(invalid("Expected canonical AUDIO payload"));
            };
            let compute = context.compute_session().map_err(invalid)?;
            let execution = compute.execution_context(&context).map_err(invalid)?;
            let indices = if self.list {
                0..selector
            } else {
                selector..selector + 1
            };
            let mut published = Vec::new();
            let result = (|| {
                let mut columns = [Vec::new(), Vec::new(), Vec::new()];
                for index in indices {
                    check(&context)?;
                    let payloads =
                        pad_segment(images, audio, length, index, compute.backend(), &execution)
                            .map_err(|error| failure(&context, error.message))?;
                    for (column, payload) in columns.iter_mut().zip(payloads) {
                        check(&context)?;
                        let handle = context
                            .handle_store()
                            .publish(payload, &context.cancellation)
                            .map_err(|error| failure(&context, error))?;
                        published.push(handle.clone());
                        column.push(NativeValue::Handle { value: handle });
                    }
                }
                let outputs = columns
                    .into_iter()
                    .map(|mut values| {
                        if self.list {
                            Ok(NativeValue::List { values })
                        } else {
                            values
                                .pop()
                                .ok_or_else(|| invalid("Segment output was not prepared"))
                        }
                    })
                    .collect::<Result<Vec<_>, NativeNodeFailure>>()?;
                let outcome = NativeNodeOutcome::Values {
                    outputs,
                    ui: None,
                    effects: Vec::new(),
                };
                check(&context)?;
                outcome.validate().map_err(invalid)?;
                Ok(outcome)
            })();
            if result.is_err() {
                let cancellation = sim_types::CancellationToken::default();
                let mut cleanup_error = None;
                for handle in published.iter().rev() {
                    if let Err(error) = context.handle_store().revoke(handle, &cancellation) {
                        cleanup_error.get_or_insert(error);
                    }
                }
                if let Some(error) = cleanup_error {
                    return Err(failure(
                        &context,
                        format!("Padding output rollback failed: {error}"),
                    ));
                }
            }
            result
        })
    }
}

fn integer(
    inputs: &BTreeMap<String, NativeValue>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, NativeNodeFailure> {
    let value = match inputs.get(name) {
        Some(NativeValue::Primitive {
            value: NativePrimitive::Integer(value),
        }) => u64::try_from(*value).ok(),
        Some(NativeValue::Primitive {
            value: NativePrimitive::UnsignedInteger(value),
        }) => Some(*value),
        _ => None,
    };
    value
        .filter(|value| (minimum..=maximum).contains(value))
        .ok_or_else(|| {
            invalid(format!(
                "{name} must be an integer from {minimum} through {maximum}"
            ))
        })
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
        _ => Err(invalid(format!("{name} requires a scalar {kind} handle"))),
    }
}
fn invalid(error: impl std::fmt::Display) -> NativeNodeFailure {
    NativeNodeFailure {
        code: "invalid_wandancer_padding".to_owned(),
        message: error.to_string(),
        kind: NativeNodeFailureKind::Failure,
        retryable: false,
    }
}
fn failure(context: &NativeNodeContext, error: impl std::fmt::Display) -> NativeNodeFailure {
    if context.cancellation.is_cancelled() {
        NativeNodeFailure {
            code: "interrupted".to_owned(),
            message: "WanDancer padding cancelled".to_owned(),
            kind: NativeNodeFailureKind::Interrupted,
            retryable: true,
        }
    } else {
        invalid(error)
    }
}
fn check(context: &NativeNodeContext) -> Result<(), NativeNodeFailure> {
    context
        .cancellation
        .check()
        .map_err(|error| failure(context, error))
}

fn source_segment_count(
    audio_frames: u64,
    sample_rate: u32,
    segment_length: u64,
) -> Result<u64, NativeNodeFailure> {
    let duration = audio_frames as f64 / f64::from(sample_rate);
    if duration <= 0.2 {
        return Ok(0);
    }
    let count = ((duration - 0.2) / (segment_length as f64 / 30.0)).trunc();
    if !count.is_finite() || count >= u64::MAX as f64 {
        return Err(invalid("Audio duration exceeds segment arithmetic"));
    }
    (count as u64)
        .checked_add(1)
        .ok_or_else(|| invalid("Segment count overflowed"))
}

fn visit_source_positions(
    images: u64,
    total_frames: u64,
    segment_length: u64,
    segment_index: u64,
    mut visit: impl FnMut(u64, u64) -> Result<(), NativeNodeFailure>,
    mut check: impl FnMut() -> Result<(), NativeNodeFailure>,
) -> Result<(), NativeNodeFailure> {
    if total_frames == 0 || images == 0 {
        return Ok(());
    }
    let frame_interval = total_frames as f64 / images as f64;
    let segment_count = (total_frames as f64 / segment_length as f64).ceil() as u64;
    let end_for = |index: u64| -> Result<u64, NativeNodeFailure> {
        if index == segment_count.saturating_sub(1) {
            total_frames
                .checked_sub(
                    segment_length
                        .checked_mul(index)
                        .ok_or_else(|| invalid("Segment offset overflowed"))?,
                )
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| invalid("Segment end overflowed"))
        } else {
            Ok(segment_length - 1)
        }
    };
    // The source excludes each prior segment's appended final keyframe from consumption.
    let mut consumed = 0u64;
    for previous in 0..segment_index {
        let end = end_for(previous)?;
        let mut count = 0u64;
        while (count as f64) * frame_interval < end as f64 - frame_interval {
            check()?;
            count = count
                .checked_add(1)
                .ok_or_else(|| invalid("Image count overflowed"))?;
        }
        consumed = consumed
            .checked_add(count)
            .ok_or_else(|| invalid("Image count overflowed"))?;
    }
    let end = end_for(segment_index)?;
    let mut count = 0u64;
    while (count as f64) * frame_interval < end as f64 - frame_interval {
        check()?;
        let position = (frame_interval * count as f64).ceil() as u64;
        let source = consumed
            .checked_add(count)
            .ok_or_else(|| invalid("Image index overflowed"))?;
        if source < images && position < segment_length {
            visit(position, source)?;
        }
        count = count
            .checked_add(1)
            .ok_or_else(|| invalid("Image count overflowed"))?;
    }
    let source = consumed
        .checked_add(count)
        .ok_or_else(|| invalid("Image index overflowed"))?;
    if source < images && end < segment_length {
        visit(end, source)?;
    }
    Ok(())
}

fn pad_segment(
    images: &ImageTensor,
    audio: &NativeAudioPayload,
    segment_length: u64,
    segment_index: u64,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<[NativeStoredPayload; 3], NativeNodeFailure> {
    context.check().map_err(invalid)?;
    let (batch, height, width, channels) = images.dimensions().map_err(invalid)?;
    let [_, _, audio_frames] = audio.waveform().descriptor().shape() else {
        return Err(invalid("Invalid AUDIO shape"));
    };
    let total_frames = source_segment_count(*audio_frames, audio.sample_rate(), segment_length)?
        .checked_mul(segment_length)
        .ok_or_else(|| invalid("Total frame count overflowed"))?;
    let mut sources = backend
        .workspace_vec(context, usize::try_from(segment_length).map_err(invalid)?)
        .map_err(invalid)?;
    for _ in 0..segment_length {
        sources.try_push(None::<u64>).map_err(invalid)?;
    }
    visit_source_positions(
        batch,
        total_frames,
        segment_length,
        segment_index,
        |position, source| {
            *sources
                .get_mut(usize::try_from(position).map_err(invalid)?)
                .ok_or_else(|| invalid("Keyframe position exceeds segment"))? = Some(source);
            Ok(())
        },
        || context.check().map_err(invalid),
    )?;
    let pixels_per_frame = height
        .checked_mul(width)
        .ok_or_else(|| invalid("Frame dimensions overflowed"))?;
    let values_per_frame = pixels_per_frame
        .checked_mul(channels)
        .ok_or_else(|| invalid("Frame channels overflowed"))?;
    let output_pixels = pixels_per_frame
        .checked_mul(segment_length)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("Mask dimensions overflowed"))?;
    let output_values = values_per_frame
        .checked_mul(segment_length)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("Image dimensions overflowed"))?;
    let mut pixels = backend
        .workspace_vec(context, output_values)
        .map_err(invalid)?;
    let mut mask = backend
        .workspace_vec(context, output_pixels)
        .map_err(invalid)?;
    let input = images.as_f32_slice().map_err(invalid)?;
    for source in sources.iter() {
        context.check().map_err(invalid)?;
        let start = source
            .map(|index| {
                index
                    .checked_mul(values_per_frame)
                    .ok_or_else(|| invalid("Image offset overflowed"))
            })
            .transpose()?;
        for offset in 0..values_per_frame {
            if offset % 4096 == 0 {
                context.check().map_err(invalid)?;
            }
            let value = match start {
                Some(start) => *input
                    .get(
                        usize::try_from(
                            start
                                .checked_add(offset)
                                .ok_or_else(|| invalid("Image offset overflowed"))?,
                        )
                        .map_err(invalid)?,
                    )
                    .ok_or_else(|| invalid("Image offset exceeds storage"))?,
                None => 0.0,
            };
            pixels.try_push(value).map_err(invalid)?;
        }
        for offset in 0..pixels_per_frame {
            if offset % 4096 == 0 {
                context.check().map_err(invalid)?;
            }
            mask.try_push(if source.is_some() { 1.0 } else { 0.0 })
                .map_err(invalid)?;
        }
    }
    let image = ImageTensor::from_f32(
        backend,
        context,
        segment_length,
        height,
        width,
        channels,
        &pixels,
    )
    .map_err(invalid)?;
    let mask = ImageTensor::from_f32(backend, context, segment_length, height, width, 1, &mask)
        .map_err(invalid)?;
    let duration = *audio_frames as f64 / f64::from(audio.sample_rate());
    let segment_duration = segment_length as f64 / 30.0;
    let start_time = segment_index as f64 * segment_duration;
    let start = ((start_time * f64::from(audio.sample_rate())).trunc() as u64).min(*audio_frames);
    let end = (((start_time + segment_duration).min(duration) * f64::from(audio.sample_rate()))
        .trunc() as u64)
        .min(*audio_frames)
        .max(start);
    let sliced = narrow_function_exact_native(
        audio.waveform(),
        2,
        i64::try_from(start).map_err(invalid)?,
        end - start,
        context.cancellation,
    )
    .map_err(invalid)?;
    let waveform = contiguous_with_context_exact_native(
        backend,
        &sliced,
        MemoryFormatReference::Layout(sim_tensor::Layout::Contiguous),
        context,
    )
    .map_err(invalid)?;
    let audio = NativeAudioPayload::checked_cancellable(
        waveform,
        audio.sample_rate(),
        context.cancellation,
    )
    .map_err(invalid)?;
    Ok([
        NativeStoredPayload::Tensor(Arc::new(
            NativeTensorPayload::from_image(NativeTensorRole::Image, image).map_err(invalid)?,
        )),
        NativeStoredPayload::Tensor(Arc::new(
            NativeTensorPayload::from_image(NativeTensorRole::Mask, mask).map_err(invalid)?,
        )),
        NativeStoredPayload::Audio(Arc::new(audio)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn wandancer_padding_rows_preserve_source_contracts() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/nodes/image-video-comfy-node-0762/fixture.json"
        )))?;
        let rows = fixture["nodes"].as_array().ok_or("rows absent")?;
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
                return Err("executable binding absent".into());
            };
            descriptor.validate_exact_schema_v2()?;
            assert_eq!(Some(feature_id.as_str()), row["feature_id"].as_str());
            assert_eq!(
                serde_json::to_value(built_in_source_schema(node.class_type())?)?,
                row["schema"]
            );
            assert_eq!(
                *descriptor,
                serde_json::from_str::<NativeNodeDescriptor>(&serde_json::to_string(descriptor)?)?
            );
            assert_eq!(descriptor.cache, NativeCachePolicy::InputIdentity);
            assert_eq!(descriptor.effect, NativeEffectClass::Pure);
            assert!(!descriptor.output_node);
            assert!(descriptor.inputs.iter().all(|input| input.required
                && !input.hidden
                && !input.lazy
                && input.cardinality == NativePortCardinality::Scalar));
            assert!(
                descriptor
                    .outputs
                    .iter()
                    .all(|output| output.is_list == node.class_type().ends_with("List"))
            );
        }
        Ok(())
    }
}
