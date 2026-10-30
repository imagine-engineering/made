use crate::{
    CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId, ExecutionContext, ResizeCrop,
    ResizeMode, ResizeSpec, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation,
    ViewAccess,
};

#[derive(Clone, Debug)]
pub struct Rgb8ImageTensor {
    tensor: Tensor,
}

impl Rgb8ImageTensor {
    pub fn from_tensor(tensor: Tensor) -> Result<Self, TensorError> {
        let descriptor = tensor.descriptor();
        if descriptor.device() != DeviceId::CPU {
            return Err(TensorError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual: descriptor.device(),
            });
        }
        if descriptor.dtype() != DType::U8 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::U8,
                actual: descriptor.dtype(),
            });
        }
        if !descriptor.is_contiguous()? {
            return Err(TensorError::Faulted {
                reason: "RGB8 image tensors must be contiguous".to_owned(),
            });
        }
        match descriptor.shape() {
            [_, _, 3] => Ok(Self { tensor }),
            shape if shape.len() != 3 => Err(TensorError::Faulted {
                reason: format!(
                    "RGB8 image tensors must have HWC rank three, got rank {}",
                    shape.len()
                ),
            }),
            [_, _, channels] => Err(TensorError::Faulted {
                reason: format!("RGB8 image tensors require three channels, got {channels}"),
            }),
            _ => Err(TensorError::Faulted {
                reason: "RGB8 image tensor shape validation failed".to_owned(),
            }),
        }
    }

    pub fn from_logical_chw(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        input: &Tensor,
    ) -> Result<Self, TensorError> {
        context.check()?;
        let descriptor = input.descriptor();
        if descriptor.device() != DeviceId::CPU {
            return Err(TensorError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual: descriptor.device(),
            });
        }
        if !matches!(descriptor.dtype(), DType::F32 | DType::U8) {
            return Err(TensorError::UnsupportedCapability {
                operation: "zed.cpu.rgb8-image-from-logical-chw".to_owned(),
                device: descriptor.device(),
                reason: format!(
                    "logical CHW image conversion supports F32 and U8, got {:?}",
                    descriptor.dtype()
                ),
            });
        }
        if descriptor.stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: descriptor.stream(),
            });
        }
        let [channels, height, width] = descriptor.shape() else {
            return Err(TensorError::Faulted {
                reason: format!(
                    "logical RGB image tensors must have CHW rank three, got rank {}",
                    descriptor.rank()
                ),
            });
        };
        if *channels != 3 {
            return Err(TensorError::Faulted {
                reason: format!("logical RGB image tensors require three channels, got {channels}"),
            });
        }

        let height_usize = usize::try_from(*height).map_err(|_| TensorError::ShapeOverflow)?;
        let width_usize = usize::try_from(*width).map_err(|_| TensorError::ShapeOverflow)?;
        let byte_count = height_usize
            .checked_mul(width_usize)
            .and_then(|pixel_count| pixel_count.checked_mul(3))
            .ok_or(TensorError::ShapeOverflow)?;
        let mut interleaved = backend.workspace_vec(context, byte_count)?;
        for height_index in 0..*height {
            for width_index in 0..*width {
                if width_index & 0x3ff == 0 {
                    context.check()?;
                }
                for channel_index in 0..3_u64 {
                    interleaved.try_push(logical_channel_to_u8(
                        input,
                        &[channel_index, height_index, width_index],
                    )?)?;
                }
            }
        }
        context.check()?;
        let output_descriptor = TensorDescriptor::contiguous(
            vec![*height, *width, 3],
            DType::U8,
            DeviceId::CPU,
            descriptor.stream(),
        )?;
        let (output, _) = backend.upload_bytes(output_descriptor, &interleaved, context)?;
        Self::from_tensor(output)
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dimensions(&self) -> Result<(u64, u64), TensorError> {
        match self.tensor.descriptor().shape() {
            [height, width, 3] => Ok((*height, *width)),
            _ => Err(TensorError::Faulted {
                reason: "RGB8 image tensor shape changed after validation".to_owned(),
            }),
        }
    }

    pub fn height(&self) -> Result<u64, TensorError> {
        self.dimensions().map(|(height, _)| height)
    }

    pub fn width(&self) -> Result<u64, TensorError> {
        self.dimensions().map(|(_, width)| width)
    }

    pub fn as_u8_slice(&self) -> Result<&[u8], TensorError> {
        self.tensor.contiguous_bytes()
    }
}

fn logical_channel_to_u8(input: &Tensor, indices: &[u64]) -> Result<u8, TensorError> {
    match input
        .descriptor()
        .dtype()
        .decode_scalar(input.element_bytes(indices)?)?
    {
        DecodedScalar::Unsigned(value) if input.descriptor().dtype() == DType::U8 => {
            u8::try_from(value).map_err(|_| TensorError::InvalidNumeric {
                reason: format!("decoded U8 image channel is out of range: {value}"),
            })
        }
        DecodedScalar::Real(value) if input.descriptor().dtype() == DType::F32 => {
            let scaled = (value as f32) * 255.0;
            Ok((scaled.trunc() as i64) as u8)
        }
        _ => Err(TensorError::Faulted {
            reason: "canonical scalar decoder returned an unexpected RGB channel class".to_owned(),
        }),
    }
}

#[derive(Clone, Debug)]
pub struct ImageTensor {
    tensor: Tensor,
}

impl ImageTensor {
    pub fn from_f32(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        batch: u64,
        height: u64,
        width: u64,
        channels: u64,
        values: &[f32],
    ) -> Result<Self, TensorError> {
        let descriptor = image_descriptor(batch, height, width, channels, context.stream)?;
        let (tensor, _) = backend.upload_f32(descriptor, values, context)?;
        Self::from_tensor(tensor)
    }

    pub fn from_tensor(tensor: Tensor) -> Result<Self, TensorError> {
        let descriptor = tensor.descriptor();
        if descriptor.rank() != 4
            || descriptor.dtype() != DType::F32
            || descriptor.device() != DeviceId::CPU
            || !descriptor.is_contiguous()?
        {
            return Err(TensorError::Faulted {
                reason: "IMAGE tensors must be contiguous CPU F32 tensors in BHWC order".to_owned(),
            });
        }
        let channels = descriptor
            .shape()
            .get(3)
            .copied()
            .ok_or(TensorError::ShapeOverflow)?;
        if !matches!(channels, 1 | 3 | 4) {
            return Err(TensorError::Faulted {
                reason: format!("IMAGE channel count must be 1, 3, or 4, got {channels}"),
            });
        }
        Ok(Self { tensor })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dimensions(&self) -> Result<(u64, u64, u64, u64), TensorError> {
        let shape = self.tensor.descriptor().shape();
        match shape {
            [batch, height, width, channels] => Ok((*batch, *height, *width, *channels)),
            _ => Err(TensorError::Faulted {
                reason: "IMAGE tensor rank changed after validation".to_owned(),
            }),
        }
    }

    #[cfg(test)]
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, TensorError> {
        let source_values = self.as_f32_slice()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(source_values.len())
            .map_err(|error| TensorError::Faulted {
                reason: format!("image tensor allocation failed: {error}"),
            })?;
        values.extend_from_slice(source_values);
        Ok(values)
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], TensorError> {
        bytemuck::try_cast_slice(self.tensor.contiguous_bytes()?).map_err(|error| {
            TensorError::Faulted {
                reason: format!("IMAGE tensor F32 storage is invalid: {error}"),
            }
        })
    }

    pub fn numpy_image_bytes(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<CpuWorkspaceVec<u8>, TensorError> {
        context.check()?;
        if self.tensor.descriptor().stream() != context.stream {
            return Err(TensorError::StreamMismatch {
                expected: context.stream,
                actual: self.tensor.descriptor().stream(),
            });
        }
        let values = self.as_f32_slice()?;
        let mut bytes = backend.workspace_vec(context, values.len())?;
        for (index, value) in values.iter().copied().enumerate() {
            if index.is_multiple_of(256) {
                context.check()?;
            }
            // The qualified Linux x86_64 NumPy wheel uses a signed i32 conversion
            // sentinel on invalid input; Rust's saturating cast differs on positive overflow.
            let scaled = value * 255.0;
            let integer =
                if scaled.is_finite() && (-2_147_483_648.0..2_147_483_648.0).contains(&scaled) {
                    scaled.trunc() as i32
                } else {
                    i32::MIN
                };
            bytes.try_push(integer as u8)?;
        }
        context.check()?;
        Ok(bytes)
    }

    pub fn source_compatible_u8_crop(
        &self,
        left: u64,
        top: u64,
        width: u64,
        height: u64,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, TensorError> {
        context.check()?;
        let (batch, input_height, input_width, channels) = self.dimensions()?;
        if left >= input_width || top >= input_height || width == 0 || height == 0 {
            return Err(TensorError::Faulted {
                reason: "source-compatible image crop is outside the input bounds".to_owned(),
            });
        }
        let output_width = width.min(input_width - left);
        let output_height = height.min(input_height - top);
        let element_count = batch
            .checked_mul(output_height)
            .and_then(|value| value.checked_mul(output_width))
            .and_then(|value| value.checked_mul(channels))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(TensorError::ShapeOverflow)?;
        let input = self.as_f32_slice()?;
        let mut output = backend.workspace_vec(context, element_count)?;
        for batch_index in 0..batch {
            for output_y in 0..output_height {
                context.check()?;
                for output_x in 0..output_width {
                    for channel in 0..channels {
                        let input_index = image_offset(
                            batch_index,
                            top + output_y,
                            left + output_x,
                            channel,
                            input_height,
                            input_width,
                            channels,
                        )?;
                        let value = input.get(input_index).copied().ok_or_else(|| {
                            TensorError::Faulted {
                                reason: "source-compatible crop index exceeded IMAGE storage"
                                    .to_owned(),
                            }
                        })?;
                        let quantized = (value * 255.0).trunc() as i64 as u8;
                        output.try_push(f32::from(quantized) / 255.0)?;
                    }
                }
            }
        }
        context.check()?;
        Self::from_f32(
            backend,
            context,
            batch,
            output_height,
            output_width,
            channels,
            &output,
        )
    }

    pub fn invert(
        &self,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, TensorError> {
        let (tensor, _) = backend.unary(
            UnaryOperation::InvertUnitInterval,
            &self.tensor,
            self.tensor.descriptor().clone(),
            context,
        )?;
        Self::from_tensor(tensor)
    }

    pub fn resize(
        &self,
        width: u64,
        height: u64,
        mode: ResizeMode,
        crop: ResizeCrop,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, TensorError> {
        if width == 0 && height == 0 {
            context.check()?;
            return Ok(self.clone());
        }
        let (batch, input_height, input_width, channels) = self.dimensions()?;
        let stream = self.tensor.descriptor().stream();
        if input_width == 0 || input_height == 0 {
            return Err(TensorError::Faulted {
                reason: "cannot resize an IMAGE tensor with a zero spatial dimension".to_owned(),
            });
        }
        let width = if width == 0 {
            proportional_dimension(input_width, height, input_height)?
        } else {
            width
        };
        let height = if height == 0 {
            proportional_dimension(input_height, width, input_width)?
        } else {
            height
        };
        let nchw_input = self.tensor.view(
            TensorDescriptor::channels_last(
                vec![batch, channels, input_height, input_width],
                DType::F32,
                DeviceId::CPU,
                stream,
            )?,
            ViewAccess::ReadOnly,
        )?;
        let output_descriptor = TensorDescriptor::channels_last(
            vec![batch, channels, height, width],
            DType::F32,
            DeviceId::CPU,
            stream,
        )?;
        let (nchw_output, _) = backend.resize(
            ResizeSpec {
                width,
                height,
                mode,
                crop,
                antialias: false,
                align_corners: false,
            },
            &nchw_input,
            output_descriptor,
            context,
        )?;
        let bhwc_output = nchw_output.view(
            image_descriptor(batch, height, width, channels, stream)?,
            ViewAccess::Writable,
        )?;
        Self::from_tensor(bhwc_output)
    }

    pub fn alpha_composite_rgba8(
        background: &[u8],
        overlay: &[u8],
        width: u64,
        height: u64,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, TensorError> {
        context.check()?;
        let pixel_count = width
            .checked_mul(height)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(TensorError::ShapeOverflow)?;
        let rgba_bytes = pixel_count
            .checked_mul(4)
            .ok_or(TensorError::ShapeOverflow)?;
        if background.len() != rgba_bytes || overlay.len() != rgba_bytes {
            return Err(TensorError::Faulted {
                reason: "RGBA alpha composition requires exact contiguous image storage".to_owned(),
            });
        }
        let output_elements = pixel_count
            .checked_mul(3)
            .ok_or(TensorError::ShapeOverflow)?;
        let mut output = backend.workspace_vec(context, output_elements)?;
        for pixel in 0..pixel_count {
            if pixel & 0x3ff == 0 {
                context.check()?;
            }
            let offset = pixel.checked_mul(4).ok_or(TensorError::ShapeOverflow)?;
            let foreground_alpha =
                u32::from(*overlay.get(offset + 3).ok_or(TensorError::ShapeOverflow)?);
            let background_alpha = u32::from(
                *background
                    .get(offset + 3)
                    .ok_or(TensorError::ShapeOverflow)?,
            );
            let inverse_foreground_alpha = 255_u32.saturating_sub(foreground_alpha);
            let output_alpha_numerator = foreground_alpha
                .checked_mul(255)
                .and_then(|value| {
                    background_alpha
                        .checked_mul(inverse_foreground_alpha)
                        .and_then(|background| value.checked_add(background))
                })
                .ok_or(TensorError::ShapeOverflow)?;
            for channel in 0..3 {
                let foreground = u32::from(
                    *overlay
                        .get(offset + channel)
                        .ok_or(TensorError::ShapeOverflow)?,
                );
                let background = u32::from(
                    *background
                        .get(offset + channel)
                        .ok_or(TensorError::ShapeOverflow)?,
                );
                let premultiplied = foreground
                    .checked_mul(foreground_alpha)
                    .and_then(|value| value.checked_mul(255))
                    .and_then(|value| {
                        background
                            .checked_mul(background_alpha)
                            .and_then(|value| value.checked_mul(inverse_foreground_alpha))
                            .and_then(|background| value.checked_add(background))
                    })
                    .ok_or(TensorError::ShapeOverflow)?;
                let value = premultiplied
                    .checked_add(output_alpha_numerator / 2)
                    .ok_or(TensorError::ShapeOverflow)?
                    .checked_div(output_alpha_numerator)
                    .unwrap_or(0);
                output.try_push(
                    f32::from(
                        u8::try_from(value).map_err(|_| TensorError::InvalidNumeric {
                            reason: "RGBA alpha composition exceeded the U8 channel range"
                                .to_owned(),
                        })?,
                    ) / 255.0,
                )?;
            }
        }
        context.check()?;
        Self::from_f32(backend, context, 1, height, width, 3, &output)
    }
}

fn image_offset(
    batch: u64,
    y: u64,
    x: u64,
    channel: u64,
    height: u64,
    width: u64,
    channels: u64,
) -> Result<usize, TensorError> {
    batch
        .checked_mul(height)
        .and_then(|value| value.checked_add(y))
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_add(x))
        .and_then(|value| value.checked_mul(channels))
        .and_then(|value| value.checked_add(channel))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(TensorError::ShapeOverflow)
}

fn image_descriptor(
    batch: u64,
    height: u64,
    width: u64,
    channels: u64,
    stream: crate::StreamId,
) -> Result<TensorDescriptor, TensorError> {
    TensorDescriptor::contiguous(
        vec![batch, height, width, channels],
        DType::F32,
        DeviceId::CPU,
        stream,
    )
}

fn proportional_dimension(
    input_dimension: u64,
    requested_dimension: u64,
    input_requested_dimension: u64,
) -> Result<u64, TensorError> {
    let value = (input_dimension as f64) * (requested_dimension as f64)
        / (input_requested_dimension as f64);
    if !value.is_finite() || value > u64::MAX as f64 {
        return Err(TensorError::ShapeOverflow);
    }
    Ok((value.round_ties_even() as u64).max(1))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{CancellationToken, CpuWorkspaceAuthority, StreamId};
    use serde::Deserialize;
    use std::{collections::BTreeMap, error::Error, io};

    const TEST_MEMORY_LIMIT_BYTES: u64 = 1024 * 1024;

    fn context<'a>(
        authority: &CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation,
        })
    }

    #[derive(Deserialize)]
    struct ResizeOracle {
        inputs: BTreeMap<String, ResizeOracleInput>,
        cases: Vec<ResizeOracleCase>,
    }

    #[derive(Deserialize)]
    struct ResizeOracleInput {
        shape_bhwc: [u64; 4],
        values_flat_bhwc_f32: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct ResizeOracleCase {
        arguments: ResizeOracleArguments,
        comparison: ResizeOracleComparison,
        id: String,
        input_id: String,
        output: ResizeOracleOutput,
    }

    #[derive(Deserialize)]
    struct ResizeOracleArguments {
        crop: String,
        height: u64,
        upscale_method: String,
        width: u64,
    }

    #[derive(Deserialize)]
    struct ResizeOracleComparison {
        #[serde(default)]
        absolute_tolerance: f32,
        #[serde(default)]
        relative_tolerance: f32,
        #[serde(default)]
        alias_required: bool,
    }

    #[derive(Deserialize)]
    struct ResizeOracleOutput {
        shape_bhwc: [u64; 4],
        values_flat_bhwc_f32: Vec<f32>,
        #[serde(default)]
        values_flat_bhwc_u8: Option<Vec<u8>>,
    }

    pub(crate) fn checked_in_resize_oracle_case_results()
    -> Result<BTreeMap<&'static str, bool>, Box<dyn Error>> {
        let oracle: ResizeOracle = serde_json::from_slice(include_bytes!(
            "../../sim_test_support/fixtures/tensor_operations/image_resize_foundation.json"
        ))?;
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = context(&authority, &cancellation)?;
        let mut results = BTreeMap::new();
        for case in oracle.cases {
            let key = resize_oracle_case_key(&case.id)?;
            let input = oracle.inputs.get(&case.input_id).ok_or_else(|| {
                io::Error::other(format!(
                    "resize oracle case {} references missing input {}",
                    case.id, case.input_id
                ))
            })?;
            let [batch, height, width, channels] = input.shape_bhwc;
            let image = ImageTensor::from_f32(
                &backend,
                &context,
                batch,
                height,
                width,
                channels,
                &input.values_flat_bhwc_f32,
            )?;
            let resized = image.resize(
                case.arguments.width,
                case.arguments.height,
                resize_oracle_mode(&case.arguments.upscale_method)?,
                resize_oracle_crop(&case.arguments.crop)?,
                &backend,
                &context,
            )?;
            let actual = resized.as_f32_slice()?;
            let dimensions_match = resized.dimensions()?
                == (
                    case.output.shape_bhwc[0],
                    case.output.shape_bhwc[1],
                    case.output.shape_bhwc[2],
                    case.output.shape_bhwc[3],
                );
            let alias_matches = !case.comparison.alias_required
                || resized.tensor().storage_id() == image.tensor().storage_id();
            let values_match = actual.len() == case.output.values_flat_bhwc_f32.len()
                && actual.iter().zip(&case.output.values_flat_bhwc_f32).all(
                    |(actual, expected)| {
                        let tolerance = case.comparison.absolute_tolerance.max(
                            case.comparison.relative_tolerance * actual.abs().max(expected.abs()),
                        );
                        (actual - expected).abs() <= tolerance
                    },
                );
            let quantization_matches =
                case.output
                    .values_flat_bhwc_u8
                    .as_ref()
                    .is_none_or(|expected_bytes| {
                        actual.len() == expected_bytes.len()
                            && actual
                                .iter()
                                .zip(expected_bytes)
                                .all(|(value, byte)| *value == f32::from(*byte) / 255.0)
                    });
            results.insert(
                key,
                dimensions_match && alias_matches && values_match && quantization_matches,
            );
        }
        Ok(results)
    }

    fn resize_oracle_case_key(id: &str) -> Result<&'static str, io::Error> {
        match id {
            "nearest_exact_disabled" => Ok("resize_oracle_nearest_exact_disabled"),
            "bilinear_disabled" => Ok("resize_oracle_bilinear_disabled"),
            "area_disabled_downscale" => Ok("resize_oracle_area_disabled_downscale"),
            "bicubic_disabled" => Ok("resize_oracle_bicubic_disabled"),
            "lanczos_disabled" => Ok("resize_oracle_lanczos_disabled"),
            "bilinear_center_crop" => Ok("resize_oracle_bilinear_center_crop"),
            "nearest_exact_center_extreme" => Ok("resize_oracle_nearest_exact_center_extreme"),
            "zero_width_proportional" => Ok("resize_oracle_zero_width_proportional"),
            "zero_height_proportional" => Ok("resize_oracle_zero_height_proportional"),
            "both_dimensions_zero_identity" => Ok("resize_oracle_both_dimensions_zero_identity"),
            "proportional_minimum_one" => Ok("resize_oracle_proportional_minimum_one"),
            value => Err(io::Error::other(format!(
                "unknown checked-in resize oracle case {value}"
            ))),
        }
    }

    fn resize_oracle_mode(value: &str) -> Result<ResizeMode, io::Error> {
        match value {
            "nearest-exact" => Ok(ResizeMode::NearestExact),
            "bilinear" => Ok(ResizeMode::Bilinear),
            "area" => Ok(ResizeMode::Area),
            "bicubic" => Ok(ResizeMode::Bicubic),
            "lanczos" => Ok(ResizeMode::Lanczos),
            value => Err(io::Error::other(format!(
                "unknown resize oracle mode {value}"
            ))),
        }
    }

    fn resize_oracle_crop(value: &str) -> Result<ResizeCrop, io::Error> {
        match value {
            "disabled" => Ok(ResizeCrop::Disabled),
            "center" => Ok(ResizeCrop::Center),
            value => Err(io::Error::other(format!(
                "unknown resize oracle crop {value}"
            ))),
        }
    }

    #[test]
    fn bhwc_invert_and_resize_preserve_sim_image_layout() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = context(&authority, &cancellation)?;
        let image = ImageTensor::from_f32(
            &backend,
            &context,
            1,
            1,
            2,
            3,
            &[0.0, 0.25, 0.5, 0.75, 1.0, 0.125],
        )?;
        assert_eq!(
            image.invert(&backend, &context)?.to_f32_vec()?,
            vec![1.0, 0.75, 0.5, 0.25, 0.0, 0.875]
        );

        let resized = image.resize(
            4,
            0,
            ResizeMode::NearestExact,
            ResizeCrop::Disabled,
            &backend,
            &context,
        )?;
        assert_eq!(resized.dimensions()?, (1, 2, 4, 3));
        assert_eq!(
            resized.to_f32_vec()?,
            vec![
                0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 0.125, 0.75, 1.0, 0.125, 0.0, 0.25, 0.5,
                0.0, 0.25, 0.5, 0.75, 1.0, 0.125, 0.75, 1.0, 0.125,
            ]
        );
        Ok(())
    }

    #[test]
    fn zero_size_request_is_identity_and_aspect_rounding_matches_python() -> Result<(), TensorError>
    {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = context(&authority, &cancellation)?;
        let image = ImageTensor::from_f32(&backend, &context, 1, 2, 4, 1, &[0.0; 8])?;
        let identity = image.resize(
            0,
            0,
            ResizeMode::Bilinear,
            ResizeCrop::Center,
            &backend,
            &context,
        )?;
        assert_eq!(identity.tensor().storage_id(), image.tensor().storage_id());
        assert_eq!(
            image
                .resize(
                    5,
                    0,
                    ResizeMode::Bilinear,
                    ResizeCrop::Disabled,
                    &backend,
                    &context,
                )?
                .dimensions()?,
            (1, 2, 5, 1)
        );
        Ok(())
    }

    #[test]
    fn image_allocations_share_the_injected_backend_budget() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(32)?;
        let context = context(&authority, &cancellation)?;
        let image = ImageTensor::from_f32(
            &backend,
            &context,
            1,
            1,
            2,
            3,
            &[0.0, 0.25, 0.5, 0.75, 1.0, 0.125],
        )?;
        assert_eq!(backend.memory_snapshot().current_bytes, 32);
        assert!(matches!(
            image.invert(&backend, &context),
            Err(TensorError::AllocationFailed { requested: 32, .. })
        ));
        assert_eq!(backend.memory_snapshot().peak_bytes, 32);
        drop(image);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
        Ok(())
    }

    #[test]
    fn image_operations_preserve_and_enforce_the_execution_stream() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let stream_context = ExecutionContext {
            stream: crate::StreamId::new(7),
            scratch: authority.authorize_workspace(0)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let image = ImageTensor::from_f32(&backend, &stream_context, 1, 1, 1, 3, &[0.0, 0.5, 1.0])?;
        assert_eq!(image.tensor().descriptor().stream(), stream_context.stream);
        let inverted = image.invert(&backend, &stream_context)?;
        assert_eq!(
            inverted.tensor().descriptor().stream(),
            stream_context.stream
        );
        let resized = image.resize(
            2,
            2,
            ResizeMode::NearestExact,
            ResizeCrop::Disabled,
            &backend,
            &stream_context,
        )?;
        assert_eq!(
            resized.tensor().descriptor().stream(),
            stream_context.stream
        );

        let default_context = context(&authority, &cancellation)?;
        let before = backend.memory_snapshot().current_bytes;
        assert!(matches!(
            image.invert(&backend, &default_context),
            Err(TensorError::StreamMismatch { .. })
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn val_memory_001_image_upload_cancellation_releases_reserved_storage() {
        let values = vec![0.5_f32; 2048 * 2048];
        let cancellation = CancellationToken::default();
        let (backend, authority) =
            CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024).expect("CPU backend");
        let context = context(&authority, &cancellation).expect("workspace authorization");
        let result = std::thread::scope(|scope| {
            let upload = scope
                .spawn(|| ImageTensor::from_f32(&backend, &context, 1, 2048, 2048, 1, &values));
            let mut observed_reservation = false;
            while !upload.is_finished() {
                if backend.memory_snapshot().current_bytes > 0 {
                    observed_reservation = true;
                    break;
                }
                std::thread::yield_now();
            }
            assert!(
                observed_reservation,
                "upload completed before reservation was observable"
            );
            cancellation.cancel();
            upload.join().expect("upload thread")
        });
        assert!(matches!(result, Err(TensorError::Cancelled)));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    }

    #[test]
    fn rgb8_adapter_interleaves_non_contiguous_chw_and_matches_torchvision_quantization()
    -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let scratch = authority.authorize_workspace(6)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let source_descriptor = TensorDescriptor::contiguous(
            vec![1, 2, 3],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (source, _) = backend.upload_f32(
            source_descriptor,
            &[0.0, 0.5, 1.0, 1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0],
            &context,
        )?;
        let logical_chw = source.view(
            TensorDescriptor::new_strided(
                vec![3, 1, 2],
                vec![1, 6, 3],
                0,
                DType::F32,
                crate::Layout::Strided,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            ViewAccess::ReadOnly,
        )?;

        let image = Rgb8ImageTensor::from_logical_chw(&backend, &context, &logical_chw)?;

        assert_eq!(image.dimensions()?, (1, 2));
        assert_eq!(image.height()?, 1);
        assert_eq!(image.width()?, 2);
        assert_eq!(image.as_u8_slice()?, &[0, 127, 255, 1, 2, 3]);
        assert_eq!(image.tensor().descriptor().shape(), &[1, 2, 3]);
        assert!(image.tensor().descriptor().is_contiguous()?);
        Ok(())
    }

    #[test]
    fn rgb8_adapter_preserves_u8_channels() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let scratch = authority.authorize_workspace(6)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let descriptor = TensorDescriptor::contiguous(
            vec![3, 1, 2],
            DType::U8,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let (source, _) = backend.upload_bytes(descriptor, &[10, 20, 30, 40, 50, 60], &context)?;

        let image = Rgb8ImageTensor::from_logical_chw(&backend, &context, &source)?;

        assert_eq!(image.as_u8_slice()?, &[10, 30, 50, 20, 40, 60]);
        Ok(())
    }

    #[test]
    fn source_compatible_crop_quantizes_like_the_pillow_adapter() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let scratch = authority.authorize_workspace(TEST_MEMORY_LIMIT_BYTES)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let image = ImageTensor::from_f32(
            &backend,
            &context,
            1,
            2,
            3,
            1,
            &[0.0, 0.1, 0.5, 0.9, 1.0, 1.1],
        )?;
        let cropped = image.source_compatible_u8_crop(1, 0, 2, 2, &backend, &context)?;
        assert_eq!(cropped.dimensions()?, (1, 2, 2, 1));
        assert_eq!(
            cropped.to_f32_vec()?,
            vec![25.0 / 255.0, 127.0 / 255.0, 1.0, 24.0 / 255.0]
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        assert!(
            image
                .source_compatible_u8_crop(3, 0, 1, 1, &backend, &context)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rgb8_adapter_rejects_invalid_rank_channel_dtype_and_device() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = context(&authority, &cancellation)?;

        let (invalid_rank, _) = backend.upload_f32(
            TensorDescriptor::contiguous(vec![3, 2], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
            &[0.0; 6],
            &context,
        )?;
        assert!(matches!(
            Rgb8ImageTensor::from_logical_chw(&backend, &context, &invalid_rank),
            Err(TensorError::Faulted { .. })
        ));

        let (invalid_channels, _) = backend.upload_f32(
            TensorDescriptor::contiguous(
                vec![1, 1, 1],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0.0],
            &context,
        )?;
        assert!(matches!(
            Rgb8ImageTensor::from_logical_chw(&backend, &context, &invalid_channels),
            Err(TensorError::Faulted { .. })
        ));

        let (invalid_dtype, _) = backend.upload_bytes(
            TensorDescriptor::contiguous(
                vec![3, 1, 1],
                DType::I32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0; 12],
            &context,
        )?;
        assert!(matches!(
            Rgb8ImageTensor::from_logical_chw(&backend, &context, &invalid_dtype),
            Err(TensorError::UnsupportedCapability { .. })
        ));

        let device = DeviceId::new(sim_types::DeviceKind::Cuda, 0);
        let device_descriptor =
            TensorDescriptor::contiguous(vec![3, 1, 1], DType::F32, device, StreamId::DEFAULT)?;
        let invalid_device = Tensor {
            id: invalid_channels.id,
            mutation: invalid_channels.mutation,
            descriptor: device_descriptor,
            storage: invalid_channels.storage,
            access: ViewAccess::ReadOnly,
        };
        assert!(matches!(
            Rgb8ImageTensor::from_logical_chw(&backend, &context, &invalid_device),
            Err(TensorError::DeviceMismatch { actual, .. }) if actual == device
        ));
        Ok(())
    }

    #[test]
    fn rgb8_adapter_cancellation_does_not_publish_or_reserve_output() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = context(&authority, &cancellation)?;
        let (source, _) = backend.upload_f32(
            TensorDescriptor::contiguous(
                vec![3, 1, 1],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &[0.0, 0.5, 1.0],
            &context,
        )?;
        let before = backend.memory_snapshot().current_bytes;
        cancellation.cancel();

        assert!(matches!(
            Rgb8ImageTensor::from_logical_chw(&backend, &context, &source),
            Err(TensorError::Cancelled)
        ));
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn rgb8_adapter_roundtrips_through_task66_native_to_tensor() -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let scratch = authority.authorize_workspace(24)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
        let values = [
            0.0,
            3.0 / 255.0,
            1.0,
            1.0 / 255.0,
            4.0 / 255.0,
            127.0 / 255.0,
        ];
        let (source, _) = backend.upload_f32(
            TensorDescriptor::contiguous(
                vec![3, 1, 2],
                DType::F32,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            &values,
            &context,
        )?;
        let image = Rgb8ImageTensor::from_logical_chw(&backend, &context, &source)?;
        let (height, width) = image.dimensions()?;
        let roundtrip =
            crate::generated_external_tensor_kernel_01::to_tensor_with_context_exact_native(
                &backend,
                image.as_u8_slice()?,
                height,
                width,
                3,
                StreamId::DEFAULT,
                &context,
            )?;

        let actual: &[f32] =
            bytemuck::try_cast_slice(roundtrip.contiguous_bytes()?).map_err(|error| {
                io::Error::other(format!("invalid roundtrip F32 storage: {error:?}"))
            })?;
        assert_eq!(actual, values);
        Ok(())
    }

    #[test]
    fn numpy_image_cast_matches_qualified_source_boundaries() -> Result<(), Box<dyn Error>> {
        #[derive(serde::Deserialize)]
        struct CastCase {
            input_bits: Vec<u32>,
            output: Vec<u8>,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            cast_cases: Vec<CastCase>,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sim_test_support/fixtures/providers/common/source_image_upload_linux_x86_64.json"
        )))?;
        assert_eq!(fixture.cast_cases.len(), 6);
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(TEST_MEMORY_LIMIT_BYTES)?,
            &cancellation,
        );
        for case in fixture.cast_cases {
            let values: Vec<f32> = case.input_bits.into_iter().map(f32::from_bits).collect();
            let image = ImageTensor::from_f32(
                &backend,
                &context,
                1,
                1,
                u64::try_from(values.len())?,
                1,
                &values,
            )?;
            let before = backend.memory_snapshot().current_bytes;
            let bytes = image.numpy_image_bytes(&backend, &context)?;
            assert_eq!(&*bytes, case.output);
            assert_eq!(
                image
                    .as_f32_slice()?
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            drop(bytes);
            assert_eq!(backend.memory_snapshot().current_bytes, before);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn rgba_alpha_composition_matches_pillow_straight_alpha_rounding() -> Result<(), TensorError> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(TEST_MEMORY_LIMIT_BYTES)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(12)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let image = ImageTensor::alpha_composite_rgba8(
            &[10, 20, 30, 128],
            &[200, 100, 50, 64],
            1,
            1,
            &backend,
            &context,
        )?;
        assert_eq!(
            image.as_f32_slice()?,
            &[86.0 / 255.0, 52.0 / 255.0, 38.0 / 255.0]
        );
        drop(image);
        let transparent = ImageTensor::alpha_composite_rgba8(
            &[10, 20, 30, 0],
            &[200, 100, 50, 0],
            1,
            1,
            &backend,
            &context,
        )?;
        assert_eq!(transparent.as_f32_slice()?, &[0.0, 0.0, 0.0]);
        drop(transparent);
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
