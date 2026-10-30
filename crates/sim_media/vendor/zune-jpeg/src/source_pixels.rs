/*
 * Integer pixel equations derived from libjpeg-turbo 3.1.4.1's jidctint.c,
 * jdsample.c, jdcolor.c and jdmaster.c. Copyright (C) 1991-1998 Thomas G. Lane;
 * Modification developed 2002-2018 by Guido Vollbeding.
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB.
 * Copyright (C) 2009-2026 D. R. Commander.
 * Copyright (C) 2014 MIPS Technologies, Inc., California.
 * Copyright (C) 2015 Google, Inc.; 2019-2020 Arm Limited; 2013 Linaro Limited.
 * Distributed under the accompanying README.ijg terms. Source hashes are in
 * SOURCE_ALGORITHMS.json; marker, entropy and progressive decoding remain zune's.
 */
use crate::components::Components;
use crate::decoder::ColorConvert16Ptr;
use crate::errors::DecodeErrors;
use zune_core::colorspace::ColorSpace;

fn transform(values: [i64; 8]) -> [i64; 8] {
    let rotation = (values[2] + values[6]) * 4433;
    let even_second = rotation - values[6] * 15137;
    let even_third = rotation + values[2] * 6270;
    let even_first = (values[0] + values[4]) << 13;
    let even_last = (values[0] - values[4]) << 13;
    let even = [
        even_first + even_third,
        even_last + even_second,
        even_last - even_second,
        even_first - even_third,
    ];
    let odd = [values[7], values[5], values[3], values[1]];
    let first = (odd[0] + odd[3]) * -7373;
    let second = (odd[1] + odd[2]) * -20995;
    let rotation = (odd[0] + odd[2] + odd[1] + odd[3]) * 9633;
    let third = (odd[0] + odd[2]) * -16069 + rotation;
    let fourth = (odd[1] + odd[3]) * -3196 + rotation;
    let odd = [
        odd[0] * 2446 + first + third,
        odd[1] * 16819 + second + fourth,
        odd[2] * 25172 + second + third,
        odd[3] * 12299 + first + fourth,
    ];
    [
        even[0] + odd[3],
        even[1] + odd[2],
        even[2] + odd[1],
        even[3] + odd[0],
        even[3] - odd[0],
        even[2] - odd[1],
        even[1] - odd[2],
        even[0] - odd[3],
    ]
}

// Each masked range-table result is between 0 and 255.
#[allow(clippy::cast_possible_truncation)]
fn range_limit(value: i64) -> i16 {
    // Masked table lookup preserves IJG's bounded behavior for corrupt coefficients.
    match value & 1023 {
        value @ 0..=127 => (value + 128) as i16,
        128..=511 => 255,
        512..=895 => 0,
        value => (value - 896) as i16,
    }
}

pub(crate) fn idct(coefficients: &mut [i32; 64], output: &mut [i16], stride: usize) {
    let mut workspace = [0_i64; 64];
    for column in 0..8 {
        let values = core::array::from_fn(|row| i64::from(coefficients[row * 8 + column]));
        let transformed = transform(values);
        for row in 0..8 {
            workspace[row * 8 + column] = (transformed[row] + 1024) >> 11;
        }
    }
    for (row, destination) in workspace.chunks_exact(8).zip(output.chunks_mut(stride)) {
        let values = core::array::from_fn(|column| row[column]);
        for (value, destination) in transform(values)
            .into_iter()
            .zip(destination.iter_mut().take(8))
        {
            *destination = range_limit((value + 131_072) >> 18);
        }
    }
}

// Signed color arithmetic is clamped to the unsigned channel range before narrowing.
#[allow(clippy::cast_sign_loss)]
fn convert<const CHANNELS: usize, const BLUE_FIRST: bool>(
    luminance: &[i16; 16],
    blue_difference: &[i16; 16],
    red_difference: &[i16; 16],
    output: &mut [u8],
    position: &mut usize,
) {
    for (index, pixel) in output[*position..]
        .chunks_exact_mut(CHANNELS)
        .take(16)
        .enumerate()
    {
        let luminance = i32::from(luminance[index]);
        let blue = i32::from(blue_difference[index]) - 128;
        let red = i32::from(red_difference[index]) - 128;
        let red_value = (luminance + ((91881 * red + 32768) >> 16)).clamp(0, 255) as u8;
        let green_value =
            (luminance + ((-46802 * red - 22554 * blue + 32768) >> 16)).clamp(0, 255) as u8;
        let blue_value = (luminance + ((116_130 * blue + 32768) >> 16)).clamp(0, 255) as u8;
        pixel[0] = if BLUE_FIRST { blue_value } else { red_value };
        pixel[1] = green_value;
        pixel[2] = if BLUE_FIRST { red_value } else { blue_value };
        if CHANNELS == 4 {
            pixel[3] = 255;
        }
        *position += CHANNELS;
    }
}

pub(crate) fn color_converter(colorspace: ColorSpace) -> Result<ColorConvert16Ptr, DecodeErrors> {
    match colorspace {
        ColorSpace::RGB => Ok(convert::<3, false>),
        ColorSpace::RGBA => Ok(convert::<4, false>),
        ColorSpace::BGR => Ok(convert::<3, true>),
        ColorSpace::BGRA => Ok(convert::<4, true>),
        _ => Err(DecodeErrors::FormatStatic(
            "Unsupported source output colorspace",
        )),
    }
}

pub(crate) fn extend_component_edges(
    component: &mut Components,
    stripe: usize,
) -> Result<(), DecodeErrors> {
    let Some((width, height)) = component.source_dimensions else {
        return Ok(());
    };
    let stride = component.width_stride;
    if stride == 0 || width == 0 || width > stride || component.raw_coeff.len() % stride != 0 {
        return Err(DecodeErrors::FormatStatic(
            "Invalid source component dimensions",
        ));
    }
    let first_row = stripe
        .checked_mul(8)
        .and_then(|value| value.checked_mul(component.vertical_sample))
        .ok_or(DecodeErrors::FormatStatic("Source component row overflow"))?;
    let rows = height
        .saturating_sub(first_row)
        .min(component.raw_coeff.len() / stride);
    if rows == 0 {
        return Err(DecodeErrors::FormatStatic(
            "Source component stripe exceeds image",
        ));
    }
    for row in component.raw_coeff.chunks_exact_mut(stride).take(rows) {
        let edge = row[width - 1];
        row[width..].fill(edge);
    }
    let last_row = (rows - 1) * stride;
    for row in rows..component.raw_coeff.len() / stride {
        component
            .raw_coeff
            .copy_within(last_row..last_row + stride, row * stride);
    }
    Ok(())
}

pub(crate) fn horizontal(input: &[i16], _: &[i16], _: &[i16], _: &mut [i16], output: &mut [i16]) {
    for (index, pair) in output.chunks_exact_mut(2).zip(input.iter()).enumerate() {
        let (pair, current) = pair;
        let previous = input[index.saturating_sub(1)];
        let next = input[(index + 1).min(input.len() - 1)];
        pair[0] = (3 * current + previous + 1) >> 2;
        pair[1] = (3 * current + next + 2) >> 2;
    }
}

pub(crate) fn horizontal_box(
    input: &[i16],
    _: &[i16],
    _: &[i16],
    _: &mut [i16],
    output: &mut [i16],
) {
    for (pair, current) in output.chunks_exact_mut(2).zip(input) {
        pair.fill(*current);
    }
}

pub(crate) fn vertical(
    input: &[i16],
    above: &[i16],
    below: &[i16],
    _: &mut [i16],
    output: &mut [i16],
) {
    let (top, bottom) = output.split_at_mut(output.len() / 2);
    for (((current, above), below), (top, bottom)) in input
        .iter()
        .zip(above)
        .zip(below)
        .zip(top.iter_mut().zip(bottom))
    {
        *top = (3 * current + above + 1) >> 2;
        *bottom = (3 * current + below + 2) >> 2;
    }
}

// Convex interpolation of source 8-bit samples remains between 0 and 255.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn horizontal_vertical(
    input: &[i16],
    above: &[i16],
    below: &[i16],
    _: &mut [i16],
    output: &mut [i16],
) {
    let half = output.len() / 2;
    for (neighbor, destination) in [above, below].into_iter().zip(output.chunks_mut(half)) {
        for (index, pair) in destination
            .chunks_exact_mut(2)
            .enumerate()
            .take(input.len())
        {
            let previous = index.saturating_sub(1);
            let next = (index + 1).min(input.len() - 1);
            let current_sum = 3 * i32::from(input[index]) + i32::from(neighbor[index]);
            let previous_sum = 3 * i32::from(input[previous]) + i32::from(neighbor[previous]);
            let next_sum = 3 * i32::from(input[next]) + i32::from(neighbor[next]);
            pair[0] = ((3 * current_sum + previous_sum + 8) >> 4) as i16;
            pair[1] = ((3 * current_sum + next_sum + 7) >> 4) as i16;
        }
    }
}

pub(crate) fn horizontal_vertical_box(
    input: &[i16],
    _: &[i16],
    _: &[i16],
    _: &mut [i16],
    output: &mut [i16],
) {
    let half = output.len() / 2;
    for row in output.chunks_mut(half) {
        horizontal_box(input, &[], &[], &mut [], row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_idct_preserves_sparse_coefficients_and_plane_padding() {
        let mut coefficients = [0; 64];
        coefficients[0] = 128;
        coefficients[1] = -17;
        coefficients[8] = 23;
        let original = coefficients;
        let mut first = [999; 80];
        idct(&mut coefficients, &mut first, 10);
        assert_eq!(coefficients, original);
        assert!(first.chunks_exact(10).all(|row| row[8..] == [999; 2]));
        let mut repeated = [999; 80];
        idct(&mut coefficients, &mut repeated, 10);
        assert_eq!(first, repeated);
        coefficients.fill(0);
        coefficients[0] = 8;
        idct(&mut coefficients, &mut repeated, 10);
        assert!(repeated.chunks_exact(10).all(|row| row[..8] == [129; 8]));
    }

    #[test]
    fn source_rounding_keeps_distinct_horizontal_and_vertical_phases() {
        let mut output = [0; 6];
        horizontal(&[0, 2, 1], &[], &[], &mut [], &mut output);
        assert_eq!(output, [0, 1, 1, 2, 1, 1]);
        let mut output = [0; 4];
        vertical(&[2, 1], &[0, 0], &[0, 0], &mut [], &mut output);
        assert_eq!(output, [1, 1, 2, 1]);
        let mut output = [0; 8];
        horizontal_vertical_box(&[13, 47], &[], &[], &mut [], &mut output);
        assert_eq!(output, [13, 13, 47, 47, 13, 13, 47, 47]);
    }
}
