use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CpuBackend, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeLatentOperationKind {
    TonemapReinhard { multiplier: f32 },
    Sharpen { radius: u32, sigma: f32, alpha: f32 },
}

#[derive(Clone, Debug)]
pub struct NativeLatentOperation {
    kind: NativeLatentOperationKind,
    digest_sha256: String,
}

#[derive(Debug, Error)]
pub enum NativeLatentOperationError {
    #[error("invalid latent operation: {0}")]
    Invalid(&'static str),
    #[error("latent operation shape overflow")]
    Overflow,
    #[error("latent operation tensor failure: {0}")]
    Tensor(String),
}

impl From<TensorError> for NativeLatentOperationError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error.to_string())
    }
}

impl NativeLatentOperation {
    pub fn tonemap_reinhard(multiplier: f32) -> Result<Self, NativeLatentOperationError> {
        Self::checked(NativeLatentOperationKind::TonemapReinhard { multiplier })
    }

    pub fn sharpen(
        radius: u32,
        sigma: f32,
        alpha: f32,
    ) -> Result<Self, NativeLatentOperationError> {
        Self::checked(NativeLatentOperationKind::Sharpen {
            radius,
            sigma,
            alpha,
        })
    }

    pub fn checked(kind: NativeLatentOperationKind) -> Result<Self, NativeLatentOperationError> {
        let valid = match kind {
            NativeLatentOperationKind::TonemapReinhard { multiplier } => {
                multiplier.is_finite() && (0.0..=100.0).contains(&multiplier)
            }
            NativeLatentOperationKind::Sharpen {
                radius,
                sigma,
                alpha,
            } => {
                (1..=31).contains(&radius)
                    && sigma.is_finite()
                    && (0.1..=10.0).contains(&sigma)
                    && alpha.is_finite()
                    && (0.0..=5.0).contains(&alpha)
            }
        };
        if !valid {
            return Err(NativeLatentOperationError::Invalid(
                "parameter outside source widget range",
            ));
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.latent-operation.v1");
        match kind {
            NativeLatentOperationKind::TonemapReinhard { multiplier } => {
                hasher.update([1]);
                hasher.update(multiplier.to_bits().to_le_bytes());
            }
            NativeLatentOperationKind::Sharpen {
                radius,
                sigma,
                alpha,
            } => {
                hasher.update([2]);
                hasher.update(radius.to_le_bytes());
                hasher.update(sigma.to_bits().to_le_bytes());
                hasher.update(alpha.to_bits().to_le_bytes());
            }
        }
        Ok(Self {
            kind,
            digest_sha256: format!("{:x}", hasher.finalize()),
        })
    }

    pub fn validate(&self) -> Result<(), NativeLatentOperationError> {
        if Self::checked(self.kind)?.digest_sha256 != self.digest_sha256 {
            return Err(NativeLatentOperationError::Invalid(
                "operation identity changed",
            ));
        }
        Ok(())
    }

    pub fn kind(&self) -> NativeLatentOperationKind {
        self.kind
    }
    pub fn digest_sha256(&self) -> &str {
        &self.digest_sha256
    }
    pub fn resident_bytes(&self) -> Result<u64, NativeLatentOperationError> {
        u64::try_from(std::mem::size_of::<Self>() + self.digest_sha256.capacity())
            .map_err(|_| NativeLatentOperationError::Overflow)
    }

    pub fn apply(
        &self,
        latent: &Tensor,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeLatentOperationError> {
        self.validate()?;
        context.check()?;
        let shape = latent.descriptor().shape();
        if shape.len() < 3
            || (matches!(self.kind, NativeLatentOperationKind::Sharpen { .. }) && shape.len() != 4)
        {
            return Err(NativeLatentOperationError::Invalid(
                "source operation requires image-like samples",
            ));
        }
        let batch = usize::try_from(*shape.first().ok_or(NativeLatentOperationError::Overflow)?)
            .map_err(|_| NativeLatentOperationError::Overflow)?;
        let channels = usize::try_from(*shape.get(1).ok_or(NativeLatentOperationError::Overflow)?)
            .map_err(|_| NativeLatentOperationError::Overflow)?;
        if batch == 0 || channels == 0 {
            return Err(NativeLatentOperationError::Invalid(
                "empty batch or channels",
            ));
        }
        let count = shape.iter().skip(2).try_fold(1_usize, |total, value| {
            total
                .checked_mul(
                    usize::try_from(*value).map_err(|_| NativeLatentOperationError::Overflow)?,
                )
                .ok_or(NativeLatentOperationError::Overflow)
        })?;
        if count == 0 {
            return Err(NativeLatentOperationError::Invalid(
                "empty spatial or temporal dimension",
            ));
        }
        let input = tensor_to_f32(backend, latent, context)
            .map_err(|error| NativeLatentOperationError::Tensor(error.to_string()))?;
        let mut output = backend.workspace_vec::<f32>(context, input.len())?;
        for _ in 0..input.len() {
            output.try_push(0.0)?;
        }
        match self.kind {
            NativeLatentOperationKind::TonemapReinhard { multiplier } => {
                let mut magnitudes = backend.workspace_vec::<f32>(
                    context,
                    batch
                        .checked_mul(count)
                        .ok_or(NativeLatentOperationError::Overflow)?,
                )?;
                for sample in 0..batch {
                    for position in 0..count {
                        context.check()?;
                        let sum = (0..channels)
                            .map(|channel| {
                                let value = input[(sample * channels + channel) * count + position];
                                value * value
                            })
                            .sum::<f32>();
                        magnitudes.try_push(sum.sqrt() + 1e-10)?;
                    }
                }
                for sample in 0..batch {
                    context.check()?;
                    let values = &magnitudes[sample * count..(sample + 1) * count];
                    let mean = values.iter().sum::<f32>() / count as f32;
                    let variance = values
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f32>()
                        / (count as f32 - 1.0);
                    let top = (variance.sqrt() * 5.0 + mean) * multiplier;
                    for position in 0..count {
                        let magnitude = values[position];
                        let scaled = magnitude * (1.0 / top);
                        let new_magnitude = scaled / (scaled + 1.0) * top;
                        for channel in 0..channels {
                            let index = (sample * channels + channel) * count + position;
                            output[index] = input[index] / magnitude * new_magnitude;
                        }
                    }
                }
            }
            NativeLatentOperationKind::Sharpen {
                radius,
                sigma,
                alpha,
            } => {
                let height =
                    usize::try_from(shape[2]).map_err(|_| NativeLatentOperationError::Overflow)?;
                let width =
                    usize::try_from(shape[3]).map_err(|_| NativeLatentOperationError::Overflow)?;
                let radius =
                    usize::try_from(radius).map_err(|_| NativeLatentOperationError::Overflow)?;
                if radius >= height || radius >= width {
                    return Err(NativeLatentOperationError::Invalid(
                        "reflect pad exceeds source dimensions",
                    ));
                }
                let kernel_size = radius
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or(NativeLatentOperationError::Overflow)?;
                let mut kernel = backend.workspace_vec::<f32>(
                    context,
                    kernel_size
                        .checked_mul(kernel_size)
                        .ok_or(NativeLatentOperationError::Overflow)?,
                )?;
                for y in 0..kernel_size {
                    for x in 0..kernel_size {
                        let fy = -1.0 + 2.0 * y as f32 / (kernel_size - 1) as f32;
                        let fx = -1.0 + 2.0 * x as f32 / (kernel_size - 1) as f32;
                        kernel.try_push((-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp())?;
                    }
                }
                let sum = kernel.iter().sum::<f32>();
                for value in kernel.iter_mut() {
                    *value = *value / sum * alpha * -10.0;
                }
                let negative_sum = kernel.iter().sum::<f32>();
                kernel[radius * kernel_size + radius] += 1.0 - negative_sum;
                let mut luminance = backend.workspace_vec::<f32>(
                    context,
                    batch
                        .checked_mul(count)
                        .ok_or(NativeLatentOperationError::Overflow)?,
                )?;
                for sample in 0..batch {
                    for position in 0..count {
                        let sum = (0..channels)
                            .map(|channel| {
                                let value = input[(sample * channels + channel) * count + position];
                                value * value
                            })
                            .sum::<f32>();
                        luminance.try_push(sum.sqrt() + 1e-6)?;
                    }
                }
                for sample in 0..batch {
                    for channel in 0..channels {
                        for y in 0..height {
                            context.check()?;
                            for x in 0..width {
                                let mut value = 0.0_f32;
                                for kernel_y in 0..kernel_size {
                                    for kernel_x in 0..kernel_size {
                                        let source_y = reflect(
                                            y as i64 + kernel_y as i64 - radius as i64,
                                            height,
                                        );
                                        let source_x = reflect(
                                            x as i64 + kernel_x as i64 - radius as i64,
                                            width,
                                        );
                                        let position = source_y * width + source_x;
                                        let index =
                                            (sample * channels + channel) * count + position;
                                        value += kernel[kernel_y * kernel_size + kernel_x]
                                            * input[index]
                                            / luminance[sample * count + position];
                                    }
                                }
                                let position = y * width + x;
                                output[(sample * channels + channel) * count + position] =
                                    value * luminance[sample * count + position];
                            }
                        }
                    }
                }
            }
        }
        tensor_from_f32(backend, shape, &output, context)
            .map_err(|error| NativeLatentOperationError::Tensor(error.to_string()))
    }
}

fn reflect(index: i64, size: usize) -> usize {
    let period = 2 * (size - 1) as i64;
    let index = index.rem_euclid(period);
    index.min(period - index) as usize
}
