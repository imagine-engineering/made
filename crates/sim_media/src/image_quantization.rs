use std::collections::BTreeMap;

use sim_tensor::{CpuBackend, ExecutionContext, ImageTensor, TensorError};
use sim_types::CancellationToken;
use thiserror::Error;

const MAX_QUANTIZATION_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeImageDither {
    None,
    FloydSteinberg,
    Bayer2,
    Bayer4,
    Bayer8,
    Bayer16,
}

#[derive(Debug, Error)]
pub enum NativeImageQuantizationError {
    #[error("image quantization input is invalid")]
    InvalidInput,
    #[error("image quantization exceeded its bounded allocation")]
    TooLarge,
    #[error("image quantization was cancelled")]
    Cancelled,
    #[error("image tensor conversion failed: {0}")]
    Tensor(#[from] TensorError),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Rgb {
    red: u8,
    green: u8,
    blue: u8,
}

#[derive(Clone, Copy, Debug)]
struct HistogramEntry {
    color: Rgb,
    count: u64,
}

#[derive(Debug)]
struct ColorBox {
    entries: Vec<HistogramEntry>,
}

impl ColorBox {
    fn population(&self) -> u64 {
        self.entries.iter().map(|entry| entry.count).sum()
    }

    fn channel_ranges(&self) -> [u8; 3] {
        let mut minimum = [u8::MAX; 3];
        let mut maximum = [u8::MIN; 3];
        for entry in &self.entries {
            let channels = [entry.color.red, entry.color.green, entry.color.blue];
            for channel in 0..3 {
                minimum[channel] = minimum[channel].min(channels[channel]);
                maximum[channel] = maximum[channel].max(channels[channel]);
            }
        }
        [
            maximum[0].saturating_sub(minimum[0]),
            maximum[1].saturating_sub(minimum[1]),
            maximum[2].saturating_sub(minimum[2]),
        ]
    }

    fn split(mut self) -> Option<(Self, Self)> {
        if self.entries.len() < 2 {
            return None;
        }
        let ranges = self.channel_ranges();
        let channel = ranges
            .iter()
            .enumerate()
            .max_by_key(|(channel, range)| (**range, std::cmp::Reverse(*channel)))
            .map(|(channel, _)| channel)?;
        self.entries.sort_by_key(|entry| match channel {
            0 => (entry.color.red, entry.color.green, entry.color.blue),
            1 => (entry.color.green, entry.color.red, entry.color.blue),
            _ => (entry.color.blue, entry.color.red, entry.color.green),
        });
        let population = self.population();
        let mut prefix = 0u64;
        let mut split_at = 1usize;
        for (index, entry) in self.entries.iter().enumerate() {
            prefix = prefix.saturating_add(entry.count);
            if prefix.saturating_mul(2) >= population {
                split_at = (index + 1).min(self.entries.len() - 1);
                break;
            }
        }
        let right = self.entries.split_off(split_at);
        Some((self, Self { entries: right }))
    }

    fn representative(&self) -> Rgb {
        let mut sums = [0u64; 3];
        let mut population = 0u64;
        for entry in &self.entries {
            population = population.saturating_add(entry.count);
            sums[0] = sums[0].saturating_add(u64::from(entry.color.red) * entry.count);
            sums[1] = sums[1].saturating_add(u64::from(entry.color.green) * entry.count);
            sums[2] = sums[2].saturating_add(u64::from(entry.color.blue) * entry.count);
        }
        let divisor = population.max(1);
        Rgb {
            red: (sums[0] / divisor).min(u64::from(u8::MAX)) as u8,
            green: (sums[1] / divisor).min(u64::from(u8::MAX)) as u8,
            blue: (sums[2] / divisor).min(u64::from(u8::MAX)) as u8,
        }
    }
}

pub fn quantize_image_tensor(
    image: &ImageTensor,
    colors: u16,
    dither: NativeImageDither,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<ImageTensor, NativeImageQuantizationError> {
    context.check()?;
    let (batch, height, width, channels) = image.dimensions()?;
    if channels != 3 || colors == 0 || colors > 256 {
        return Err(NativeImageQuantizationError::InvalidInput);
    }
    let frame_bytes = checked_rgb_bytes(width, height)?;
    let output_values = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(NativeImageQuantizationError::TooLarge)?;
    if output_values
        .checked_mul(4)
        .is_none_or(|bytes| bytes > MAX_QUANTIZATION_BYTES)
    {
        return Err(NativeImageQuantizationError::TooLarge);
    }
    let source = image.as_f32_slice()?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(output_values)
        .map_err(|_| NativeImageQuantizationError::TooLarge)?;
    for batch_index in 0..batch {
        check_cancel(context.cancellation)?;
        let start = usize::try_from(batch_index)
            .ok()
            .and_then(|index| index.checked_mul(frame_bytes))
            .ok_or(NativeImageQuantizationError::TooLarge)?;
        let end = start
            .checked_add(frame_bytes)
            .ok_or(NativeImageQuantizationError::TooLarge)?;
        let frame = source
            .get(start..end)
            .ok_or(NativeImageQuantizationError::InvalidInput)?;
        let mut rgb = Vec::new();
        rgb.try_reserve_exact(frame_bytes)
            .map_err(|_| NativeImageQuantizationError::TooLarge)?;
        rgb.extend(
            frame
                .iter()
                .map(|value| (value * 255.0).trunc() as i64 as u8),
        );
        let quantized = quantize_rgb8(&rgb, width, height, colors, dither, context.cancellation)?;
        values.extend(quantized.into_iter().map(|value| f32::from(value) / 255.0));
    }
    context.check()?;
    ImageTensor::from_f32(backend, context, batch, height, width, 3, &values).map_err(Into::into)
}

pub fn quantize_rgb8(
    pixels: &[u8],
    width: u64,
    height: u64,
    colors: u16,
    dither: NativeImageDither,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, NativeImageQuantizationError> {
    check_cancel(cancellation)?;
    let byte_count = checked_rgb_bytes(width, height)?;
    if pixels.len() != byte_count || colors == 0 || colors > 256 {
        return Err(NativeImageQuantizationError::InvalidInput);
    }
    let palette = adaptive_palette(pixels, usize::from(colors), cancellation)?;
    match dither {
        NativeImageDither::None => map_nearest(pixels, &palette, cancellation),
        NativeImageDither::FloydSteinberg => {
            floyd_steinberg(pixels, width, height, &palette, cancellation)
        }
        NativeImageDither::Bayer2 => bayer(pixels, width, height, &palette, 2, cancellation),
        NativeImageDither::Bayer4 => bayer(pixels, width, height, &palette, 4, cancellation),
        NativeImageDither::Bayer8 => bayer(pixels, width, height, &palette, 8, cancellation),
        NativeImageDither::Bayer16 => bayer(pixels, width, height, &palette, 16, cancellation),
    }
}

fn adaptive_palette(
    pixels: &[u8],
    colors: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<Rgb>, NativeImageQuantizationError> {
    let mut histogram = BTreeMap::<Rgb, u64>::new();
    for (index, pixel) in pixels.chunks_exact(3).enumerate() {
        if index & 0x3fff == 0 {
            check_cancel(cancellation)?;
        }
        let color = rgb_from_slice(pixel)?;
        let count = histogram.entry(color).or_default();
        *count = count.saturating_add(1);
    }
    let entries = histogram
        .into_iter()
        .map(|(color, count)| HistogramEntry { color, count })
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err(NativeImageQuantizationError::InvalidInput);
    }
    let mut boxes = vec![ColorBox { entries }];
    while boxes.len() < colors {
        check_cancel(cancellation)?;
        let Some(index) = boxes
            .iter()
            .enumerate()
            .filter(|(_, color_box)| color_box.entries.len() > 1)
            .max_by_key(|(_, color_box)| {
                let range = color_box.channel_ranges().into_iter().max().unwrap_or(0);
                (range, color_box.population())
            })
            .map(|(index, _)| index)
        else {
            break;
        };
        let color_box = boxes.remove(index);
        let Some((left, right)) = color_box.split() else {
            break;
        };
        boxes.push(left);
        boxes.push(right);
    }
    Ok(boxes
        .iter()
        .map(ColorBox::representative)
        .collect::<Vec<_>>())
}

fn map_nearest(
    pixels: &[u8],
    palette: &[Rgb],
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, NativeImageQuantizationError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(pixels.len())
        .map_err(|_| NativeImageQuantizationError::TooLarge)?;
    for (index, pixel) in pixels.chunks_exact(3).enumerate() {
        if index & 0x3fff == 0 {
            check_cancel(cancellation)?;
        }
        let pixel = rgb_from_slice(pixel)?;
        let nearest = nearest_color(
            [
                f32::from(pixel.red),
                f32::from(pixel.green),
                f32::from(pixel.blue),
            ],
            palette,
        )?;
        output.extend([nearest.red, nearest.green, nearest.blue]);
    }
    Ok(output)
}

fn floyd_steinberg(
    pixels: &[u8],
    width: u64,
    height: u64,
    palette: &[Rgb],
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, NativeImageQuantizationError> {
    let mut working = pixels
        .iter()
        .map(|value| f32::from(*value))
        .collect::<Vec<_>>();
    let mut output = vec![0u8; pixels.len()];
    let width = usize::try_from(width).map_err(|_| NativeImageQuantizationError::TooLarge)?;
    let height = usize::try_from(height).map_err(|_| NativeImageQuantizationError::TooLarge)?;
    for y in 0..height {
        check_cancel(cancellation)?;
        for x in 0..width {
            let offset = pixel_offset(x, y, width)?;
            let current_slice = working
                .get(offset..offset + 3)
                .ok_or(NativeImageQuantizationError::InvalidInput)?;
            let current = [current_slice[0], current_slice[1], current_slice[2]];
            let nearest = nearest_color(current, palette)?;
            let quantized = [nearest.red, nearest.green, nearest.blue];
            output
                .get_mut(offset..offset + 3)
                .ok_or(NativeImageQuantizationError::InvalidInput)?
                .copy_from_slice(&quantized);
            for channel in 0..3 {
                let error = current[channel] - f32::from(quantized[channel]);
                diffuse(
                    &mut working,
                    x + 1,
                    y,
                    width,
                    height,
                    channel,
                    error * 7.0 / 16.0,
                )?;
                if x > 0 {
                    diffuse(
                        &mut working,
                        x - 1,
                        y + 1,
                        width,
                        height,
                        channel,
                        error * 3.0 / 16.0,
                    )?;
                }
                diffuse(
                    &mut working,
                    x,
                    y + 1,
                    width,
                    height,
                    channel,
                    error * 5.0 / 16.0,
                )?;
                diffuse(
                    &mut working,
                    x + 1,
                    y + 1,
                    width,
                    height,
                    channel,
                    error / 16.0,
                )?;
            }
        }
    }
    Ok(output)
}

fn diffuse(
    working: &mut [f32],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    channel: usize,
    error: f32,
) -> Result<(), NativeImageQuantizationError> {
    if x >= width || y >= height {
        return Ok(());
    }
    let offset = pixel_offset(x, y, width)?
        .checked_add(channel)
        .ok_or(NativeImageQuantizationError::TooLarge)?;
    let value = working
        .get_mut(offset)
        .ok_or(NativeImageQuantizationError::InvalidInput)?;
    *value = (*value + error).clamp(0.0, 255.0);
    Ok(())
}

fn bayer(
    pixels: &[u8],
    width: u64,
    height: u64,
    palette: &[Rgb],
    order: usize,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, NativeImageQuantizationError> {
    let matrix = normalized_bayer(order)?;
    let spread = 512.0 / palette.len() as f32;
    let width = usize::try_from(width).map_err(|_| NativeImageQuantizationError::TooLarge)?;
    let height = usize::try_from(height).map_err(|_| NativeImageQuantizationError::TooLarge)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(pixels.len())
        .map_err(|_| NativeImageQuantizationError::TooLarge)?;
    for y in 0..height {
        check_cancel(cancellation)?;
        for x in 0..width {
            let offset = pixel_offset(x, y, width)?;
            let matrix_offset = (y % order)
                .checked_mul(order)
                .and_then(|value| value.checked_add(x % order))
                .ok_or(NativeImageQuantizationError::TooLarge)?;
            let adjustment = spread
                * matrix
                    .get(matrix_offset)
                    .copied()
                    .ok_or(NativeImageQuantizationError::InvalidInput)?
                + 0.5;
            let pixel = rgb_from_slice(
                pixels
                    .get(offset..offset + 3)
                    .ok_or(NativeImageQuantizationError::InvalidInput)?,
            )?;
            let adjusted = [
                (f32::from(pixel.red) + adjustment)
                    .clamp(0.0, 255.0)
                    .trunc(),
                (f32::from(pixel.green) + adjustment)
                    .clamp(0.0, 255.0)
                    .trunc(),
                (f32::from(pixel.blue) + adjustment)
                    .clamp(0.0, 255.0)
                    .trunc(),
            ];
            let nearest = nearest_color(adjusted, palette)?;
            output.extend([nearest.red, nearest.green, nearest.blue]);
        }
    }
    Ok(output)
}

fn normalized_bayer(order: usize) -> Result<Vec<f32>, NativeImageQuantizationError> {
    if !matches!(order, 2 | 4 | 8 | 16) {
        return Err(NativeImageQuantizationError::InvalidInput);
    }
    let mut matrix = vec![0.0f32];
    let mut size = 1usize;
    while size < order {
        let next_size = size
            .checked_mul(2)
            .ok_or(NativeImageQuantizationError::TooLarge)?;
        let next_len = next_size
            .checked_mul(next_size)
            .ok_or(NativeImageQuantizationError::TooLarge)?;
        let q = next_len as f32;
        let mut next = vec![0.0f32; next_len];
        for y in 0..size {
            for x in 0..size {
                let source = y
                    .checked_mul(size)
                    .and_then(|value| value.checked_add(x))
                    .ok_or(NativeImageQuantizationError::TooLarge)?;
                let value = matrix
                    .get(source)
                    .copied()
                    .ok_or(NativeImageQuantizationError::InvalidInput)?
                    * q;
                for (offset_y, offset_x, adjustment) in [
                    (0, 0, -1.5),
                    (0, size, 0.5),
                    (size, 0, 1.5),
                    (size, size, -0.5),
                ] {
                    let index = y
                        .checked_add(offset_y)
                        .and_then(|row| row.checked_mul(next_size))
                        .and_then(|row| row.checked_add(x))
                        .and_then(|column| column.checked_add(offset_x))
                        .ok_or(NativeImageQuantizationError::TooLarge)?;
                    let target = next
                        .get_mut(index)
                        .ok_or(NativeImageQuantizationError::InvalidInput)?;
                    *target = (value + adjustment) / q;
                }
            }
        }
        matrix = next;
        size = next_size;
    }
    Ok(matrix)
}

fn nearest_color(values: [f32; 3], palette: &[Rgb]) -> Result<Rgb, NativeImageQuantizationError> {
    palette
        .iter()
        .copied()
        .min_by(|left, right| {
            color_distance(values, *left)
                .total_cmp(&color_distance(values, *right))
                .then_with(|| left.cmp(right))
        })
        .ok_or(NativeImageQuantizationError::InvalidInput)
}

fn rgb_from_slice(values: &[u8]) -> Result<Rgb, NativeImageQuantizationError> {
    let [red, green, blue] = values else {
        return Err(NativeImageQuantizationError::InvalidInput);
    };
    Ok(Rgb {
        red: *red,
        green: *green,
        blue: *blue,
    })
}

fn color_distance(values: [f32; 3], color: Rgb) -> f32 {
    let red = values[0] - f32::from(color.red);
    let green = values[1] - f32::from(color.green);
    let blue = values[2] - f32::from(color.blue);
    red * red + green * green + blue * blue
}

fn checked_rgb_bytes(width: u64, height: u64) -> Result<usize, NativeImageQuantizationError> {
    let bytes = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(NativeImageQuantizationError::TooLarge)?;
    if width == 0 || height == 0 || bytes > MAX_QUANTIZATION_BYTES {
        return Err(NativeImageQuantizationError::TooLarge);
    }
    Ok(bytes)
}

fn pixel_offset(x: usize, y: usize, width: usize) -> Result<usize, NativeImageQuantizationError> {
    y.checked_mul(width)
        .and_then(|value| value.checked_add(x))
        .and_then(|value| value.checked_mul(3))
        .ok_or(NativeImageQuantizationError::TooLarge)
}

fn check_cancel(cancellation: &CancellationToken) -> Result<(), NativeImageQuantizationError> {
    cancellation
        .check()
        .map_err(|_| NativeImageQuantizationError::Cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_palette_and_all_dither_profiles_are_bounded_and_deterministic() {
        let pixels = [
            0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0,
        ];
        let cancellation = CancellationToken::default();
        let none = quantize_rgb8(&pixels, 3, 2, 3, NativeImageDither::None, &cancellation)
            .expect("three-color adaptive palette should quantize");
        assert_eq!(none.len(), pixels.len());
        for dither in [
            NativeImageDither::FloydSteinberg,
            NativeImageDither::Bayer2,
            NativeImageDither::Bayer4,
            NativeImageDither::Bayer8,
            NativeImageDither::Bayer16,
        ] {
            let first = quantize_rgb8(&pixels, 3, 2, 3, dither, &cancellation)
                .expect("dither profile should quantize");
            let second = quantize_rgb8(&pixels, 3, 2, 3, dither, &cancellation)
                .expect("dither profile should be repeatable");
            assert_eq!(first, second);
        }
    }

    #[test]
    fn invalid_and_cancelled_quantization_fails_closed() {
        assert!(matches!(
            quantize_rgb8(
                &[0, 0, 0],
                1,
                1,
                0,
                NativeImageDither::None,
                &CancellationToken::default()
            ),
            Err(NativeImageQuantizationError::InvalidInput)
        ));
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        assert!(matches!(
            quantize_rgb8(&[0, 0, 0], 1, 1, 1, NativeImageDither::None, &cancellation),
            Err(NativeImageQuantizationError::Cancelled)
        ));
    }
}
