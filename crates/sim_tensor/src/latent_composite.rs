//! Source-pinned CPU composition for the Task 629 latent nodes.
use crate::{
    CpuBackend, ExecutionContext, Tensor,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

#[derive(Clone, Copy, Debug)]
pub enum NativeLatentCompositeMode {
    Feather { pixels: u64 },
    Masked { resize_source: bool },
}

#[derive(Debug, Error)]
pub enum NativeLatentCompositeError {
    #[error("latent composition requires compatible rank-four F32 CPU tensors")]
    InvalidShape,
    #[error("latent composition mask shape is incompatible")]
    InvalidMask,
    #[error("latent composition coordinate or allocation overflow")]
    Overflow,
    #[error("latent composition tensor failure: {0}")]
    Tensor(String),
}

fn shape4(tensor: &Tensor) -> Result<[usize; 4], NativeLatentCompositeError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 4 {
        return Err(NativeLatentCompositeError::InvalidShape);
    }
    let mut dimensions = [0; 4];
    for (output, source) in dimensions.iter_mut().zip(shape) {
        *output = usize::try_from(*source).map_err(|_| NativeLatentCompositeError::Overflow)?;
        if *output == 0 {
            return Err(NativeLatentCompositeError::InvalidShape);
        }
    }
    Ok(dimensions)
}

fn bilinear(
    source: &[f32],
    height: usize,
    width: usize,
    y: usize,
    x: usize,
    target_height: usize,
    target_width: usize,
) -> f32 {
    let source_y = ((y as f32 + 0.5) * height as f32 / target_height as f32 - 0.5).max(0.0);
    let source_x = ((x as f32 + 0.5) * width as f32 / target_width as f32 - 0.5).max(0.0);
    let y0 = (source_y.floor() as usize).min(height - 1);
    let x0 = (source_x.floor() as usize).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let fy = source_y - y0 as f32;
    let fx = source_x - x0 as f32;
    (source[y0 * width + x0] * (1.0 - fx) + source[y0 * width + x1] * fx) * (1.0 - fy)
        + (source[y1 * width + x0] * (1.0 - fx) + source[y1 * width + x1] * fx) * fy
}

fn source_feather_reverse_slice_contains(length: usize, step: usize, position: usize) -> bool {
    let length = length as i128;
    let step = step as i128;
    let normalize = |index: i128| {
        if index < 0 {
            (length + index).max(0)
        } else {
            index.min(length)
        }
    };
    let start = normalize(length - 1 - step);
    let end = normalize(length - step);
    (position as i128) >= start && (position as i128) < end
}

/// The caller retains the destination dictionary and replaces only its sample tensor.
pub fn composite_latents_exact_native(
    backend: &CpuBackend,
    destination: &Tensor,
    source: &Tensor,
    mask: Option<&Tensor>,
    x_pixels: u64,
    y_pixels: u64,
    mode: NativeLatentCompositeMode,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeLatentCompositeError> {
    context
        .check()
        .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
    let [db, dc, dh, dw] = shape4(destination)?;
    let [sb, sc, sh, sw] = shape4(source)?;
    if (dc != sc && sc != 1)
        || (matches!(mode, NativeLatentCompositeMode::Feather { .. }) && db != sb && sb != 1)
    {
        return Err(NativeLatentCompositeError::InvalidShape);
    }
    let (oh, ow) = match mode {
        NativeLatentCompositeMode::Feather { .. } => (sh, sw),
        NativeLatentCompositeMode::Masked {
            resize_source: true,
        } => (dh, dw),
        NativeLatentCompositeMode::Masked {
            resize_source: false,
        } => (sh, sw),
    };
    let x = usize::try_from(x_pixels / 8)
        .map_err(|_| NativeLatentCompositeError::Overflow)?
        .min(dw);
    let y = usize::try_from(y_pixels / 8)
        .map_err(|_| NativeLatentCompositeError::Overflow)?
        .min(dh);
    let width = ow.min(dw - x);
    let height = oh.min(dh - y);
    let destination_values = tensor_to_f32(backend, destination, context)
        .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
    let source_values = tensor_to_f32(backend, source, context)
        .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
    let mut output = backend
        .workspace_vec::<f32>(context, destination_values.len())
        .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
    for value in destination_values.iter().copied() {
        output
            .try_push(value)
            .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
    }
    let mask_values = mask
        .map(|tensor| {
            tensor_to_f32(backend, tensor, context)
                .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))
        })
        .transpose()?;
    let mask_shape = mask.map(|tensor| tensor.descriptor().shape());
    let (mask_batch, mask_height, mask_width) = if let Some(shape) = mask_shape {
        let (batch, height, width) = match shape {
            [batch, height, width] => (*batch, *height, *width),
            [batch, one, height, width] if *one == 1 => (*batch, *height, *width),
            _ => return Err(NativeLatentCompositeError::InvalidMask),
        };
        let dimensions = [batch, height, width]
            .map(|value| usize::try_from(value).map_err(|_| NativeLatentCompositeError::Overflow))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        if dimensions.contains(&0) {
            return Err(NativeLatentCompositeError::InvalidMask);
        }
        (dimensions[0], dimensions[1], dimensions[2])
    } else {
        (0, 0, 0)
    };
    for batch in 0..db {
        context
            .check()
            .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
        let source_batch = if matches!(mode, NativeLatentCompositeMode::Masked { .. }) || sb == 1 {
            batch % sb
        } else {
            batch
        };
        for channel in 0..dc {
            let source_channel = if sc == 1 { 0 } else { channel };
            let source_plane = &source_values[(source_batch * sc + source_channel) * sh * sw
                ..(source_batch * sc + source_channel + 1) * sh * sw];
            for row in 0..height {
                context
                    .check()
                    .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))?;
                for column in 0..width {
                    let source_value = if (oh, ow) == (sh, sw) {
                        source_plane[row * sw + column]
                    } else {
                        bilinear(source_plane, sh, sw, row, column, oh, ow)
                    };
                    let weight = match mode {
                        NativeLatentCompositeMode::Feather { pixels } => {
                            let feather = usize::try_from(pixels / 8)
                                .map_err(|_| NativeLatentCompositeError::Overflow)?;
                            if feather == 0 {
                                1.0
                            } else {
                                let mut weight = 1.0_f32;
                                for step in 0..feather {
                                    let factor = (step + 1) as f32 / feather as f32;
                                    if y != 0 && row == step {
                                        weight *= factor;
                                    }
                                    if y + height < dh
                                        && source_feather_reverse_slice_contains(height, step, row)
                                    {
                                        weight *= factor;
                                    }
                                    if x != 0 && column == step {
                                        weight *= factor;
                                    }
                                    if x + width < dw
                                        && source_feather_reverse_slice_contains(
                                            width, step, column,
                                        )
                                    {
                                        weight *= factor;
                                    }
                                }
                                weight
                            }
                        }
                        NativeLatentCompositeMode::Masked { .. } => {
                            if let Some(values) = &mask_values {
                                let mask_batch = source_batch % mask_batch;
                                let plane = &values[mask_batch * mask_height * mask_width
                                    ..(mask_batch + 1) * mask_height * mask_width];
                                bilinear(plane, mask_height, mask_width, row, column, oh, ow)
                            } else {
                                1.0
                            }
                        }
                    };
                    let output_index = ((batch * dc + channel) * dh + y + row) * dw + x + column;
                    output[output_index] =
                        source_value * weight + output[output_index] * (1.0 - weight);
                }
            }
        }
    }
    tensor_from_f32(backend, destination.descriptor().shape(), &output, context)
        .map_err(|error| NativeLatentCompositeError::Tensor(error.to_string()))
}
