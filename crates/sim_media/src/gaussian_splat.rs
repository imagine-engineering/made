use crate::NativeFile3DFormat;
use flate2::{Compression, GzBuilder, read::GzDecoder};
use sim_types::CancellationToken;
use std::io::{Read, Write};
use thiserror::Error;

const SH_DC: f32 = 0.282_094_8;
const KSPLAT_HEADER_BYTES: usize = 4096;
const KSPLAT_SECTION_HEADER_BYTES: usize = 1024;
const KSPLAT_RECORD_BYTES: usize = 44;
const SPZ_MAGIC: u32 = 0x5053_474e;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaussianSplatCodecLimits {
    pub maximum_input_bytes: usize,
    pub maximum_decoded_bytes: usize,
    pub maximum_splats: usize,
    pub maximum_ply_properties: usize,
}

impl Default for GaussianSplatCodecLimits {
    fn default() -> Self {
        Self {
            maximum_input_bytes: 512 * 1024 * 1024,
            maximum_decoded_bytes: 2 * 1024 * 1024 * 1024,
            maximum_splats: 16 * 1024 * 1024,
            maximum_ply_properties: 64,
        }
    }
}

impl GaussianSplatCodecLimits {
    pub fn validate(self) -> Result<Self, GaussianSplatCodecError> {
        if self.maximum_input_bytes == 0
            || self.maximum_decoded_bytes == 0
            || self.maximum_splats == 0
            || self.maximum_ply_properties == 0
            || self.maximum_input_bytes > 2 * 1024 * 1024 * 1024
            || self.maximum_decoded_bytes > 2 * 1024 * 1024 * 1024
            || self.maximum_ply_properties > 4096
        {
            return Err(GaussianSplatCodecError::InvalidLimits);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianSplatRecord {
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation_wxyz: [f32; 4],
    pub opacity: f32,
    pub spherical_harmonics: Box<[[f32; 3]]>,
}

impl GaussianSplatRecord {
    pub fn checked(
        position: [f32; 3],
        scale: [f32; 3],
        rotation_wxyz: [f32; 4],
        opacity: f32,
        spherical_harmonics: Vec<[f32; 3]>,
    ) -> Result<Self, GaussianSplatCodecError> {
        if position.iter().any(|value| !value.is_finite())
            || scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || rotation_wxyz.iter().any(|value| !value.is_finite())
            || !opacity.is_finite()
            || !(0.0..=1.0).contains(&opacity)
            || !matches!(spherical_harmonics.len(), 1 | 4 | 9 | 16)
            || spherical_harmonics
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(GaussianSplatCodecError::InvalidRecord);
        }
        let norm = rotation_wxyz
            .iter()
            .fold(0.0_f32, |sum, value| value.mul_add(*value, sum));
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err(GaussianSplatCodecError::InvalidRecord);
        }
        let inverse_norm = norm.sqrt().recip();
        let rotation_wxyz = rotation_wxyz.map(|value| value * inverse_norm);
        Ok(Self {
            position,
            scale,
            rotation_wxyz,
            opacity,
            spherical_harmonics: spherical_harmonics.into_boxed_slice(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaussianSplatData {
    splats: Box<[GaussianSplatRecord]>,
}

impl GaussianSplatData {
    pub fn checked(
        splats: Vec<GaussianSplatRecord>,
        limits: GaussianSplatCodecLimits,
    ) -> Result<Self, GaussianSplatCodecError> {
        let limits = limits.validate()?;
        if splats.is_empty() || splats.len() > limits.maximum_splats {
            return Err(GaussianSplatCodecError::InvalidCount);
        }
        let coefficient_count = splats[0].spherical_harmonics.len();
        if splats
            .iter()
            .any(|splat| splat.spherical_harmonics.len() != coefficient_count)
        {
            return Err(GaussianSplatCodecError::InvalidRecord);
        }
        Ok(Self {
            splats: splats.into_boxed_slice(),
        })
    }

    pub fn splats(&self) -> &[GaussianSplatRecord] {
        &self.splats
    }

    pub fn spherical_harmonic_coefficient_count(&self) -> usize {
        self.splats
            .first()
            .map_or(0, |splat| splat.spherical_harmonics.len())
    }
}

#[derive(Debug, Error)]
pub enum GaussianSplatCodecError {
    #[error("gaussian splat codec limits are invalid")]
    InvalidLimits,
    #[error("gaussian splat input exceeds the configured byte limit")]
    InputTooLarge,
    #[error("gaussian splat decoded data exceeds the configured byte limit")]
    DecodedTooLarge,
    #[error("gaussian splat data is empty or exceeds the configured count limit")]
    InvalidCount,
    #[error("gaussian splat record is invalid")]
    InvalidRecord,
    #[error("gaussian splat format is invalid or unsupported")]
    InvalidFormat,
    #[error("gaussian splat input is truncated or contains trailing data")]
    InvalidLength,
    #[error("gaussian splat operation was cancelled")]
    Cancelled,
    #[error("gaussian splat allocation failed")]
    AllocationFailed,
    #[error("gaussian splat compression failed: {0}")]
    Compression(String),
}

pub fn detect_gaussian_splat_format(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
) -> Result<NativeFile3DFormat, GaussianSplatCodecError> {
    let limits = limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.maximum_input_bytes {
        return Err(GaussianSplatCodecError::InputTooLarge);
    }
    if bytes.starts_with(b"ply") {
        Ok(NativeFile3DFormat::Ply)
    } else if bytes.starts_with(&[0x1f, 0x8b]) {
        Ok(NativeFile3DFormat::Spz)
    } else if bytes.len() >= 2 && bytes[0] == 0 && bytes[1] >= 1 {
        Ok(NativeFile3DFormat::Ksplat)
    } else if bytes.len().is_multiple_of(32) {
        Ok(NativeFile3DFormat::Splat)
    } else {
        Err(GaussianSplatCodecError::InvalidFormat)
    }
}

pub fn decode_gaussian_splat(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<(NativeFile3DFormat, GaussianSplatData), GaussianSplatCodecError> {
    check_cancelled(cancellation)?;
    let format = detect_gaussian_splat_format(bytes, limits)?;
    let data = match format {
        NativeFile3DFormat::Ply => decode_ply(bytes, limits, cancellation),
        NativeFile3DFormat::Splat => decode_splat(bytes, limits, cancellation),
        NativeFile3DFormat::Ksplat => decode_ksplat(bytes, limits, cancellation),
        NativeFile3DFormat::Spz => decode_spz(bytes, limits, cancellation),
        _ => Err(GaussianSplatCodecError::InvalidFormat),
    }?;
    check_cancelled(cancellation)?;
    Ok((format, data))
}

pub fn encode_gaussian_ply(
    data: &GaussianSplatData,
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, GaussianSplatCodecError> {
    let limits = limits.validate()?;
    validate_data(data, limits)?;
    let coefficient_count = data.spherical_harmonic_coefficient_count();
    let rest_count = coefficient_count
        .checked_sub(1)
        .and_then(|count| count.checked_mul(3))
        .ok_or(GaussianSplatCodecError::InvalidRecord)?;
    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        data.splats.len()
    );
    for name in ["x", "y", "z", "nx", "ny", "nz"] {
        header.push_str(&format!("property float {name}\n"));
    }
    for index in 0..3 {
        header.push_str(&format!("property float f_dc_{index}\n"));
    }
    for index in 0..rest_count {
        header.push_str(&format!("property float f_rest_{index}\n"));
    }
    for name in [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("end_header\n");
    let floats_per_record = 17_usize
        .checked_add(rest_count)
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    let body_bytes = data
        .splats
        .len()
        .checked_mul(floats_per_record)
        .and_then(|count| count.checked_mul(4))
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    let total_bytes = header
        .len()
        .checked_add(body_bytes)
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    if total_bytes > limits.maximum_decoded_bytes {
        return Err(GaussianSplatCodecError::DecodedTooLarge);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(total_bytes)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    output.extend_from_slice(header.as_bytes());
    for (index, splat) in data.splats.iter().enumerate() {
        check_periodic(index, cancellation)?;
        write_f32s(&mut output, &splat.position);
        write_f32s(&mut output, &[0.0; 3]);
        write_f32s(&mut output, &splat.spherical_harmonics[0]);
        for channel in 0..3 {
            for coefficient in 1..coefficient_count {
                output.extend_from_slice(
                    &splat.spherical_harmonics[coefficient][channel].to_le_bytes(),
                );
            }
        }
        let opacity = splat.opacity.clamp(1.0e-6, 1.0 - 1.0e-6);
        output.extend_from_slice(&(opacity / (1.0 - opacity)).ln().to_le_bytes());
        write_f32s(
            &mut output,
            &splat.scale.map(|value| value.max(1.0e-8).ln()),
        );
        write_f32s(&mut output, &splat.rotation_wxyz);
    }
    Ok(output)
}

pub fn encode_gaussian_ksplat(
    data: &GaussianSplatData,
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, GaussianSplatCodecError> {
    let limits = limits.validate()?;
    validate_data(data, limits)?;
    let count =
        u32::try_from(data.splats.len()).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    let body_bytes = data
        .splats
        .len()
        .checked_mul(KSPLAT_RECORD_BYTES)
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    let total_bytes = KSPLAT_HEADER_BYTES
        .checked_add(KSPLAT_SECTION_HEADER_BYTES)
        .and_then(|value| value.checked_add(body_bytes))
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    if total_bytes > limits.maximum_decoded_bytes {
        return Err(GaussianSplatCodecError::DecodedTooLarge);
    }
    let mut output = vec![0_u8; KSPLAT_HEADER_BYTES + KSPLAT_SECTION_HEADER_BYTES];
    output[1] = 1;
    put_u32(&mut output, 4, 1)?;
    put_u32(&mut output, 8, 1)?;
    put_u32(&mut output, 12, count)?;
    put_u32(&mut output, 16, count)?;
    let section = KSPLAT_HEADER_BYTES;
    put_u32(&mut output, section, count)?;
    put_u32(&mut output, section + 4, count)?;
    put_u32(
        &mut output,
        section + 28,
        u32::try_from(body_bytes).map_err(|_| GaussianSplatCodecError::DecodedTooLarge)?,
    )?;
    output
        .try_reserve_exact(body_bytes)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    for (index, splat) in data.splats.iter().enumerate() {
        check_periodic(index, cancellation)?;
        write_f32s(&mut output, &splat.position);
        write_f32s(&mut output, &splat.scale);
        write_f32s(&mut output, &splat.rotation_wxyz);
        let base = splat.spherical_harmonics[0].map(|value| (value * SH_DC + 0.5).clamp(0.0, 1.0));
        output.extend(base.map(float_to_byte));
        output.push(float_to_byte(splat.opacity));
    }
    Ok(output)
}

pub fn encode_gaussian_spz(
    data: &GaussianSplatData,
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<Vec<u8>, GaussianSplatCodecError> {
    let limits = limits.validate()?;
    validate_data(data, limits)?;
    let count =
        u32::try_from(data.splats.len()).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    let raw_bytes = 16_usize
        .checked_add(
            data.splats
                .len()
                .checked_mul(19)
                .ok_or(GaussianSplatCodecError::DecodedTooLarge)?,
        )
        .ok_or(GaussianSplatCodecError::DecodedTooLarge)?;
    if raw_bytes > limits.maximum_decoded_bytes {
        return Err(GaussianSplatCodecError::DecodedTooLarge);
    }
    let mut raw = Vec::new();
    raw.try_reserve_exact(raw_bytes)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    raw.extend_from_slice(&SPZ_MAGIC.to_le_bytes());
    raw.extend_from_slice(&2_u32.to_le_bytes());
    raw.extend_from_slice(&count.to_le_bytes());
    raw.extend_from_slice(&[0, 12, 0, 0]);
    for (index, splat) in data.splats.iter().enumerate() {
        check_periodic(index, cancellation)?;
        for value in splat.position {
            let quantized = (value * 4096.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            let bytes = quantized.to_le_bytes();
            raw.extend_from_slice(&bytes[..3]);
        }
    }
    for splat in data.splats.iter() {
        raw.push(float_to_byte(splat.opacity));
    }
    let color_scale = SH_DC / 0.15;
    for splat in data.splats.iter() {
        for coefficient in splat.spherical_harmonics[0] {
            let rgb = coefficient * SH_DC + 0.5;
            raw.push(float_to_byte((rgb - 0.5) / color_scale + 0.5));
        }
    }
    for splat in data.splats.iter() {
        for scale in splat.scale {
            raw.push(
                (scale.max(1.0e-9).ln().mul_add(16.0, 160.0))
                    .round()
                    .clamp(0.0, 255.0) as u8,
            );
        }
    }
    for splat in data.splats.iter() {
        let mut rotation = splat.rotation_wxyz;
        if rotation[0] < 0.0 {
            rotation = rotation.map(|value| -value);
        }
        for value in &rotation[1..] {
            raw.push(((*value + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8);
        }
    }
    let mut encoder = GzBuilder::new()
        .mtime(0)
        .write(Vec::new(), Compression::default());
    encoder
        .write_all(&raw)
        .map_err(|error| GaussianSplatCodecError::Compression(error.to_string()))?;
    let output = encoder
        .finish()
        .map_err(|error| GaussianSplatCodecError::Compression(error.to_string()))?;
    if output.len() > limits.maximum_input_bytes {
        return Err(GaussianSplatCodecError::InputTooLarge);
    }
    Ok(output)
}

fn decode_splat(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<GaussianSplatData, GaussianSplatCodecError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(32) {
        return Err(GaussianSplatCodecError::InvalidLength);
    }
    let count = bytes.len() / 32;
    check_count(count, limits)?;
    let mut splats = Vec::new();
    splats
        .try_reserve_exact(count)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    for (index, record) in bytes.chunks_exact(32).enumerate() {
        check_periodic(index, cancellation)?;
        let position = read_f32_array::<3>(record, 0)?;
        let scale = read_f32_array::<3>(record, 12)?;
        let rgba = record
            .get(24..28)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let quaternion = record
            .get(28..32)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let rotation = [
            (f32::from(quaternion[0]) - 128.0) / 128.0,
            (f32::from(quaternion[1]) - 128.0) / 128.0,
            (f32::from(quaternion[2]) - 128.0) / 128.0,
            (f32::from(quaternion[3]) - 128.0) / 128.0,
        ];
        let rgb = [rgba[0], rgba[1], rgba[2]].map(|value| (f32::from(value) / 255.0 - 0.5) / SH_DC);
        splats.push(GaussianSplatRecord::checked(
            position,
            scale,
            rotation,
            f32::from(rgba[3]) / 255.0,
            vec![rgb],
        )?);
    }
    GaussianSplatData::checked(splats, limits)
}

fn decode_ksplat(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<GaussianSplatData, GaussianSplatCodecError> {
    if bytes.len() < KSPLAT_HEADER_BYTES || bytes[0] != 0 {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let maximum_sections =
        usize::try_from(read_u32(bytes, 4)?).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    let section_count =
        usize::try_from(read_u32(bytes, 8)?).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    let declared_count =
        usize::try_from(read_u32(bytes, 16)?).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    let level = read_u16(bytes, 20)?;
    if maximum_sections == 0
        || section_count == 0
        || section_count > maximum_sections
        || !matches!(level, 0..=2)
    {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let headers_bytes = maximum_sections
        .checked_mul(KSPLAT_SECTION_HEADER_BYTES)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    let mut base = KSPLAT_HEADER_BYTES
        .checked_add(headers_bytes)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    if base > bytes.len() {
        return Err(GaussianSplatCodecError::InvalidLength);
    }
    let mut splats = Vec::new();
    splats
        .try_reserve_exact(declared_count)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    for section_index in 0..maximum_sections {
        let section = KSPLAT_HEADER_BYTES
            .checked_add(
                section_index
                    .checked_mul(KSPLAT_SECTION_HEADER_BYTES)
                    .ok_or(GaussianSplatCodecError::InvalidLength)?,
            )
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let count = usize::try_from(read_u32(bytes, section)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let maximum_count = usize::try_from(read_u32(bytes, section + 4)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let bucket_size = usize::try_from(read_u32(bytes, section + 8)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let bucket_count = usize::try_from(read_u32(bytes, section + 12)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let block_size = read_f32(bytes, section + 16)?;
        let bucket_storage_bytes = usize::from(read_u16(bytes, section + 20)?);
        let scale_range = read_u32(bytes, section + 24)?;
        let scale_range = if scale_range == 0 {
            32_767
        } else {
            scale_range
        };
        let full_buckets = usize::try_from(read_u32(bytes, section + 32)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let partial_buckets = usize::try_from(read_u32(bytes, section + 36)?)
            .map_err(|_| GaussianSplatCodecError::InvalidCount)?;
        let sh_components = match read_u16(bytes, section + 40)? {
            0 => 0,
            1 => 9,
            2 => 24,
            3 => 45,
            _ => return Err(GaussianSplatCodecError::InvalidFormat),
        };
        if count > maximum_count || splats.len().saturating_add(count) > limits.maximum_splats {
            return Err(GaussianSplatCodecError::InvalidCount);
        }
        let (center_bytes, scale_bytes, rotation_bytes, sh_component_bytes): (
            usize,
            usize,
            usize,
            usize,
        ) = match level {
            0 => (12, 12, 16, 4),
            1 => (6, 6, 8, 2),
            2 => (6, 6, 8, 1),
            _ => return Err(GaussianSplatCodecError::InvalidFormat),
        };
        let bytes_per_splat =
            center_bytes + scale_bytes + rotation_bytes + 4 + sh_components * sh_component_bytes;
        let partial_metadata_bytes = partial_buckets
            .checked_mul(4)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let bucket_bytes = bucket_storage_bytes
            .checked_mul(bucket_count)
            .and_then(|value| value.checked_add(partial_metadata_bytes))
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let data_base = base
            .checked_add(bucket_bytes)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let section_storage_bytes = bytes_per_splat
            .checked_mul(maximum_count)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let next_base = data_base
            .checked_add(section_storage_bytes)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        if next_base > bytes.len() {
            return Err(GaussianSplatCodecError::InvalidLength);
        }
        let bucket_centers = if level == 0 {
            Vec::new()
        } else {
            if bucket_size == 0
                || bucket_count == 0
                || !block_size.is_finite()
                || block_size <= 0.0
                || scale_range == 0
            {
                return Err(GaussianSplatCodecError::InvalidFormat);
            }
            let center_start = base
                .checked_add(partial_metadata_bytes)
                .ok_or(GaussianSplatCodecError::InvalidLength)?;
            (0..bucket_count)
                .map(|index| read_f32_array::<3>(bytes, center_start + index * 12))
                .collect::<Result<Vec<_>, _>>()?
        };
        let partial_lengths = (0..partial_buckets)
            .map(|index| {
                read_u32(bytes, base + index * 4).and_then(|value| {
                    usize::try_from(value).map_err(|_| GaussianSplatCodecError::InvalidCount)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for index in 0..count {
            check_periodic(splats.len(), cancellation)?;
            let offset = data_base
                .checked_add(
                    index
                        .checked_mul(bytes_per_splat)
                        .ok_or(GaussianSplatCodecError::InvalidLength)?,
                )
                .ok_or(GaussianSplatCodecError::InvalidLength)?;
            let position = if level == 0 {
                read_f32_array::<3>(bytes, offset)?
            } else {
                let quantized = [
                    read_u16(bytes, offset)?,
                    read_u16(bytes, offset + 2)?,
                    read_u16(bytes, offset + 4)?,
                ];
                let full_splats = full_buckets
                    .checked_mul(bucket_size)
                    .ok_or(GaussianSplatCodecError::InvalidLength)?;
                let bucket_index = if index < full_splats {
                    index / bucket_size
                } else {
                    let partial_offset = index - full_splats;
                    let mut accumulated = 0_usize;
                    let mut selected = None;
                    for (partial_index, length) in partial_lengths.iter().copied().enumerate() {
                        accumulated = accumulated
                            .checked_add(length)
                            .ok_or(GaussianSplatCodecError::InvalidLength)?;
                        if partial_offset < accumulated {
                            selected = Some(full_buckets + partial_index);
                            break;
                        }
                    }
                    selected.ok_or(GaussianSplatCodecError::InvalidFormat)?
                };
                let center = bucket_centers
                    .get(bucket_index)
                    .ok_or(GaussianSplatCodecError::InvalidFormat)?;
                let factor = block_size / 2.0 / scale_range as f32;
                [0, 1, 2].map(|component| {
                    (f32::from(quantized[component]) - scale_range as f32) * factor
                        + center[component]
                })
            };
            let scale_offset = offset + center_bytes;
            let scale = if level == 0 {
                read_f32_array::<3>(bytes, scale_offset)?
            } else {
                [
                    half_to_f32(read_u16(bytes, scale_offset)?),
                    half_to_f32(read_u16(bytes, scale_offset + 2)?),
                    half_to_f32(read_u16(bytes, scale_offset + 4)?),
                ]
            };
            let rotation_offset = scale_offset + scale_bytes;
            let rotation = if level == 0 {
                read_f32_array::<4>(bytes, rotation_offset)?
            } else {
                [
                    half_to_f32(read_u16(bytes, rotation_offset)?),
                    half_to_f32(read_u16(bytes, rotation_offset + 2)?),
                    half_to_f32(read_u16(bytes, rotation_offset + 4)?),
                    half_to_f32(read_u16(bytes, rotation_offset + 6)?),
                ]
            };
            let color_offset = rotation_offset + rotation_bytes;
            let color = bytes
                .get(color_offset..color_offset + 4)
                .ok_or(GaussianSplatCodecError::InvalidLength)?;
            let sh = [color[0], color[1], color[2]]
                .map(|value| (f32::from(value) / 255.0 - 0.5) / SH_DC);
            splats.push(GaussianSplatRecord::checked(
                position,
                scale,
                rotation,
                f32::from(color[3]) / 255.0,
                vec![sh],
            )?);
        }
        base = next_base;
    }
    if splats.len() != declared_count || base != bytes.len() {
        return Err(GaussianSplatCodecError::InvalidLength);
    }
    GaussianSplatData::checked(splats, limits)
}

fn decode_spz(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<GaussianSplatData, GaussianSplatCodecError> {
    let mut decoder = GzDecoder::new(bytes);
    let mut raw = Vec::new();
    decoder
        .by_ref()
        .take(
            u64::try_from(limits.maximum_decoded_bytes)
                .map_err(|_| GaussianSplatCodecError::DecodedTooLarge)?
                .saturating_add(1),
        )
        .read_to_end(&mut raw)
        .map_err(|error| GaussianSplatCodecError::Compression(error.to_string()))?;
    if raw.len() > limits.maximum_decoded_bytes {
        return Err(GaussianSplatCodecError::DecodedTooLarge);
    }
    if read_u32(&raw, 0)? != SPZ_MAGIC {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let version = read_u32(&raw, 4)?;
    if !matches!(version, 1..=3) {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let count =
        usize::try_from(read_u32(&raw, 8)?).map_err(|_| GaussianSplatCodecError::InvalidCount)?;
    check_count(count, limits)?;
    let position_bytes = count
        .checked_mul(if version == 1 { 6 } else { 9 })
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    let rotation_bytes_per_splat = if version == 3 { 4 } else { 3 };
    let sh_components = match raw.get(12).copied() {
        Some(0) => 0,
        Some(1) => 9,
        Some(2) => 24,
        Some(3) => 45,
        _ => return Err(GaussianSplatCodecError::InvalidFormat),
    };
    let attribute_bytes = count
        .checked_mul(
            1_usize
                .checked_add(3)
                .and_then(|value| value.checked_add(3))
                .and_then(|value| value.checked_add(rotation_bytes_per_splat))
                .and_then(|value| value.checked_add(sh_components))
                .ok_or(GaussianSplatCodecError::InvalidLength)?,
        )
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    let expected = 16_usize
        .checked_add(position_bytes)
        .and_then(|value| value.checked_add(attribute_bytes))
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    if raw.len() != expected {
        return Err(GaussianSplatCodecError::InvalidLength);
    }
    let fractional_bits = *raw.get(13).ok_or(GaussianSplatCodecError::InvalidLength)?;
    if fractional_bits > 30 {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let alpha_offset = 16 + position_bytes;
    let color_offset = alpha_offset + count;
    let scale_offset = color_offset + count * 3;
    let rotation_offset = scale_offset + count * 3;
    let mut splats = Vec::new();
    splats
        .try_reserve_exact(count)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    for index in 0..count {
        check_periodic(index, cancellation)?;
        let position = if version == 1 {
            let offset = 16 + index * 6;
            [
                half_to_f32(read_u16(&raw, offset)?),
                half_to_f32(read_u16(&raw, offset + 2)?),
                half_to_f32(read_u16(&raw, offset + 4)?),
            ]
        } else {
            let offset = 16 + index * 9;
            [0, 1, 2]
                .map(|component| {
                    let start = offset + component * 3;
                    read_i24(&raw, start)
                        .map(|value| value as f32 / (1_u32 << fractional_bits) as f32)
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidLength)?
        };
        let opacity = f32::from(raw[alpha_offset + index]) / 255.0;
        let color_start = color_offset + index * 3;
        let color_scale = SH_DC / 0.15;
        let sh = [0, 1, 2].map(|component| {
            let rgb = (f32::from(raw[color_start + component]) / 255.0 - 0.5) * color_scale + 0.5;
            (rgb - 0.5) / SH_DC
        });
        let scale_start = scale_offset + index * 3;
        let scale = [0, 1, 2]
            .map(|component| (f32::from(raw[scale_start + component]) / 16.0 - 10.0).exp());
        let rotation_start = rotation_offset + index * rotation_bytes_per_splat;
        let rotation = if version == 3 {
            decode_spz_v3_rotation(&raw, rotation_start)?
        } else {
            let xyz =
                [0, 1, 2].map(|component| f32::from(raw[rotation_start + component]) / 127.5 - 1.0);
            let w = (1.0 - xyz.iter().map(|value| value * value).sum::<f32>())
                .max(0.0)
                .sqrt();
            [w, xyz[0], xyz[1], xyz[2]]
        };
        splats.push(GaussianSplatRecord::checked(
            position,
            scale,
            rotation,
            opacity,
            vec![sh],
        )?);
    }
    GaussianSplatData::checked(splats, limits)
}

fn decode_spz_v3_rotation(
    bytes: &[u8],
    offset: usize,
) -> Result<[f32; 4], GaussianSplatCodecError> {
    let mut remaining = read_u32(bytes, offset)?;
    let largest = usize::try_from((remaining >> 30) & 3)
        .map_err(|_| GaussianSplatCodecError::InvalidRecord)?;
    let mut xyzw = [0.0_f32; 4];
    let mut sum_squared = 0.0_f32;
    for component in (0..4).rev() {
        if component == largest {
            continue;
        }
        let magnitude = (remaining & 0x1ff) as f32 / 511.0 / 2.0_f32.sqrt();
        let value = if (remaining >> 9) & 1 == 1 {
            -magnitude
        } else {
            magnitude
        };
        xyzw[component] = value;
        sum_squared = value.mul_add(value, sum_squared);
        remaining >>= 10;
    }
    xyzw[largest] = (1.0 - sum_squared).max(0.0).sqrt();
    Ok([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
}

fn decode_ply(
    bytes: &[u8],
    limits: GaussianSplatCodecLimits,
    cancellation: &CancellationToken,
) -> Result<GaussianSplatData, GaussianSplatCodecError> {
    let marker = b"end_header";
    let marker_start = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or(GaussianSplatCodecError::InvalidFormat)?;
    let line_end = marker_start
        .checked_add(marker.len())
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    let body_start = if bytes.get(line_end..line_end + 2) == Some(b"\r\n") {
        line_end + 2
    } else if bytes.get(line_end) == Some(&b'\n') {
        line_end + 1
    } else {
        return Err(GaussianSplatCodecError::InvalidFormat);
    };
    let header = std::str::from_utf8(
        bytes
            .get(..marker_start)
            .ok_or(GaussianSplatCodecError::InvalidLength)?,
    )
    .map_err(|_| GaussianSplatCodecError::InvalidFormat)?;
    let mut count = None;
    let mut in_vertex = false;
    let mut properties = Vec::new();
    for line in header.lines() {
        let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["format", "binary_little_endian", "1.0"] => {}
            ["format", ..] => return Err(GaussianSplatCodecError::InvalidFormat),
            ["element", "vertex", value] => {
                count = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| GaussianSplatCodecError::InvalidCount)?,
                );
                in_vertex = true;
            }
            ["element", ..] => in_vertex = false,
            ["property", "list", ..] if in_vertex => {
                return Err(GaussianSplatCodecError::InvalidFormat);
            }
            ["property", kind, name] if in_vertex => {
                if properties.len() >= limits.maximum_ply_properties {
                    return Err(GaussianSplatCodecError::InvalidFormat);
                }
                properties.push((name.to_string(), PlyScalar::parse(kind)?));
            }
            _ => {}
        }
    }
    let count = count.ok_or(GaussianSplatCodecError::InvalidCount)?;
    check_count(count, limits)?;
    let stride = properties.iter().try_fold(0_usize, |total, (_, kind)| {
        total
            .checked_add(kind.width())
            .ok_or(GaussianSplatCodecError::InvalidLength)
    })?;
    let body_bytes = count
        .checked_mul(stride)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    if body_start.checked_add(body_bytes) != Some(bytes.len()) {
        return Err(GaussianSplatCodecError::InvalidLength);
    }
    let rest_names = properties
        .iter()
        .filter_map(|(name, _)| name.strip_prefix("f_rest_")?.parse::<usize>().ok())
        .collect::<Vec<_>>();
    let rest_count = rest_names.len();
    if rest_count % 3 != 0 {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let coefficient_count = rest_count / 3 + 1;
    if !matches!(coefficient_count, 1 | 4 | 9 | 16) {
        return Err(GaussianSplatCodecError::InvalidFormat);
    }
    let mut splats = Vec::new();
    splats
        .try_reserve_exact(count)
        .map_err(|_| GaussianSplatCodecError::AllocationFailed)?;
    for index in 0..count {
        check_periodic(index, cancellation)?;
        let record_start = body_start + index * stride;
        let mut values = std::collections::BTreeMap::new();
        let mut offset = record_start;
        for (name, kind) in &properties {
            values.insert(name.as_str(), kind.read(bytes, offset)?);
            offset += kind.width();
        }
        let value = |name: &str| values.get(name).copied();
        let position = ["x", "y", "z"]
            .map(|name| value(name).ok_or(GaussianSplatCodecError::InvalidFormat))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| GaussianSplatCodecError::InvalidFormat)?;
        let scale = if value("scale_0").is_some() {
            ["scale_0", "scale_1", "scale_2"]
                .map(|name| {
                    value(name)
                        .map(f32::exp)
                        .ok_or(GaussianSplatCodecError::InvalidFormat)
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidFormat)?
        } else {
            [0.01; 3]
        };
        let rotation = if value("rot_0").is_some() {
            ["rot_0", "rot_1", "rot_2", "rot_3"]
                .map(|name| value(name).ok_or(GaussianSplatCodecError::InvalidFormat))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidFormat)?
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };
        let opacity = value("opacity").map_or(1.0, |value| 1.0 / (1.0 + (-value).exp()));
        let dc = if value("f_dc_0").is_some() {
            ["f_dc_0", "f_dc_1", "f_dc_2"]
                .map(|name| value(name).ok_or(GaussianSplatCodecError::InvalidFormat))
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidFormat)?
        } else if value("red").is_some() {
            ["red", "green", "blue"]
                .map(|name| {
                    value(name)
                        .map(|value| (value / 255.0 - 0.5) / SH_DC)
                        .ok_or(GaussianSplatCodecError::InvalidFormat)
                })
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidFormat)?
        } else {
            [0.0; 3]
        };
        let mut harmonics = vec![[0.0; 3]; coefficient_count];
        harmonics[0] = dc;
        for channel in 0..3 {
            for coefficient in 1..coefficient_count {
                let rest_index = channel * (coefficient_count - 1) + coefficient - 1;
                harmonics[coefficient][channel] = value(&format!("f_rest_{rest_index}"))
                    .ok_or(GaussianSplatCodecError::InvalidFormat)?;
            }
        }
        splats.push(GaussianSplatRecord::checked(
            position, scale, rotation, opacity, harmonics,
        )?);
    }
    GaussianSplatData::checked(splats, limits)
}

#[derive(Clone, Copy)]
enum PlyScalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyScalar {
    fn parse(value: &str) -> Result<Self, GaussianSplatCodecError> {
        match value {
            "char" | "int8" => Ok(Self::I8),
            "uchar" | "uint8" => Ok(Self::U8),
            "short" | "int16" => Ok(Self::I16),
            "ushort" | "uint16" => Ok(Self::U16),
            "int" | "int32" => Ok(Self::I32),
            "uint" | "uint32" => Ok(Self::U32),
            "float" | "float32" => Ok(Self::F32),
            "double" | "float64" => Ok(Self::F64),
            _ => Err(GaussianSplatCodecError::InvalidFormat),
        }
    }

    const fn width(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn read(self, bytes: &[u8], offset: usize) -> Result<f32, GaussianSplatCodecError> {
        let slice = bytes
            .get(offset..offset + self.width())
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        Ok(match self {
            Self::I8 => i8::from_le_bytes([slice[0]]) as f32,
            Self::U8 => f32::from(slice[0]),
            Self::I16 => i16::from_le_bytes([slice[0], slice[1]]) as f32,
            Self::U16 => u16::from_le_bytes([slice[0], slice[1]]) as f32,
            Self::I32 => i32::from_le_bytes(
                slice
                    .try_into()
                    .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
            ) as f32,
            Self::U32 => u32::from_le_bytes(
                slice
                    .try_into()
                    .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
            ) as f32,
            Self::F32 => f32::from_le_bytes(
                slice
                    .try_into()
                    .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
            ),
            Self::F64 => f64::from_le_bytes(
                slice
                    .try_into()
                    .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
            ) as f32,
        })
    }
}

fn validate_data(
    data: &GaussianSplatData,
    limits: GaussianSplatCodecLimits,
) -> Result<(), GaussianSplatCodecError> {
    if data.splats.is_empty() || data.splats.len() > limits.maximum_splats {
        return Err(GaussianSplatCodecError::InvalidCount);
    }
    let coefficient_count = data.spherical_harmonic_coefficient_count();
    if !matches!(coefficient_count, 1 | 4 | 9 | 16)
        || data
            .splats
            .iter()
            .any(|splat| splat.spherical_harmonics.len() != coefficient_count)
    {
        return Err(GaussianSplatCodecError::InvalidRecord);
    }
    Ok(())
}

fn check_count(
    count: usize,
    limits: GaussianSplatCodecLimits,
) -> Result<(), GaussianSplatCodecError> {
    if count == 0 || count > limits.maximum_splats {
        Err(GaussianSplatCodecError::InvalidCount)
    } else {
        Ok(())
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<(), GaussianSplatCodecError> {
    cancellation
        .check()
        .map_err(|_| GaussianSplatCodecError::Cancelled)
}

fn check_periodic(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), GaussianSplatCodecError> {
    if index.is_multiple_of(4096) {
        check_cancelled(cancellation)?;
    }
    Ok(())
}

fn write_f32s(output: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_f32_array<const COUNT: usize>(
    bytes: &[u8],
    offset: usize,
) -> Result<[f32; COUNT], GaussianSplatCodecError> {
    let mut output = [0.0; COUNT];
    for (index, value) in output.iter_mut().enumerate() {
        let start = offset
            .checked_add(
                index
                    .checked_mul(4)
                    .ok_or(GaussianSplatCodecError::InvalidLength)?,
            )
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        let slice = bytes
            .get(start..start + 4)
            .ok_or(GaussianSplatCodecError::InvalidLength)?;
        *value = f32::from_le_bytes(
            slice
                .try_into()
                .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
        );
    }
    Ok(output)
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, GaussianSplatCodecError> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    Ok(u16::from_le_bytes(
        slice
            .try_into()
            .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
    ))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, GaussianSplatCodecError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    Ok(f32::from_le_bytes(
        slice
            .try_into()
            .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, GaussianSplatCodecError> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    Ok(u32::from_le_bytes(
        slice
            .try_into()
            .map_err(|_| GaussianSplatCodecError::InvalidLength)?,
    ))
}

fn read_i24(bytes: &[u8], offset: usize) -> Result<i32, GaussianSplatCodecError> {
    let slice = bytes
        .get(offset..offset + 3)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    let value = i32::from(slice[0]) | (i32::from(slice[1]) << 8) | (i32::from(slice[2]) << 16);
    Ok(if value & 0x80_0000 != 0 {
        value - 0x100_0000
    } else {
        value
    })
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), GaussianSplatCodecError> {
    let destination = bytes
        .get_mut(offset..offset + 4)
        .ok_or(GaussianSplatCodecError::InvalidLength)?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn float_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn half_to_f32(value: u16) -> f32 {
    half::f16::from_bits(value).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(coefficient_count: usize) -> Result<GaussianSplatData, GaussianSplatCodecError> {
        GaussianSplatData::checked(
            vec![GaussianSplatRecord::checked(
                [1.25, -2.5, 3.75],
                [0.125, 0.25, 0.5],
                [1.0, 0.0, 0.0, 0.0],
                0.75,
                (0..coefficient_count)
                    .map(|index| {
                        let value = index as f32 * 0.01;
                        [value + 0.1, value + 0.2, value + 0.3]
                    })
                    .collect(),
            )?],
            GaussianSplatCodecLimits::default(),
        )
    }

    #[test]
    fn binary_ply_round_trips_full_spherical_harmonics_byte_stably()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = GaussianSplatCodecLimits::default();
        let cancellation = CancellationToken::default();
        let fixture = fixture(4)?;
        let first = encode_gaussian_ply(&fixture, limits, &cancellation)?;
        let second = encode_gaussian_ply(&fixture, limits, &cancellation)?;
        assert_eq!(first, second);
        assert_eq!(
            detect_gaussian_splat_format(&first, limits)?,
            NativeFile3DFormat::Ply
        );
        let (format, decoded) = decode_gaussian_splat(&first, limits, &cancellation)?;
        assert_eq!(format, NativeFile3DFormat::Ply);
        assert_eq!(decoded.spherical_harmonic_coefficient_count(), 4);
        let decoded = &decoded.splats()[0];
        let expected = &fixture.splats()[0];
        assert_eq!(decoded.position, expected.position);
        assert_eq!(decoded.scale, expected.scale);
        assert_eq!(decoded.rotation_wxyz, expected.rotation_wxyz);
        assert!((decoded.opacity - expected.opacity).abs() <= f32::EPSILON);
        assert_eq!(decoded.spherical_harmonics, expected.spherical_harmonics);
        Ok(())
    }

    #[test]
    fn ksplat_and_spz_writers_are_deterministic_and_truthfully_reduce_to_dc()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = GaussianSplatCodecLimits::default();
        let cancellation = CancellationToken::default();
        let fixture = fixture(4)?;
        for (format, first, second) in [
            (
                NativeFile3DFormat::Ksplat,
                encode_gaussian_ksplat(&fixture, limits, &cancellation)?,
                encode_gaussian_ksplat(&fixture, limits, &cancellation)?,
            ),
            (
                NativeFile3DFormat::Spz,
                encode_gaussian_spz(&fixture, limits, &cancellation)?,
                encode_gaussian_spz(&fixture, limits, &cancellation)?,
            ),
        ] {
            assert_eq!(first, second);
            assert_eq!(detect_gaussian_splat_format(&first, limits)?, format);
            let (decoded_format, decoded) = decode_gaussian_splat(&first, limits, &cancellation)?;
            assert_eq!(decoded_format, format);
            assert_eq!(decoded.spherical_harmonic_coefficient_count(), 1);
            assert_eq!(decoded.splats().len(), 1);
        }
        Ok(())
    }

    #[test]
    fn splat_reader_rejects_malformed_oversized_and_cancelled_input()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = GaussianSplatCodecLimits {
            maximum_input_bytes: 64,
            maximum_decoded_bytes: 1024,
            maximum_splats: 1,
            maximum_ply_properties: 64,
        };
        assert!(matches!(
            detect_gaussian_splat_format(&[1, 2, 3], limits),
            Err(GaussianSplatCodecError::InvalidFormat)
        ));
        assert!(matches!(
            detect_gaussian_splat_format(&[0; 96], limits),
            Err(GaussianSplatCodecError::InputTooLarge)
        ));
        let mut bytes = Vec::new();
        write_f32s(&mut bytes, &[0.0, 0.0, 0.0]);
        write_f32s(&mut bytes, &[0.1, 0.1, 0.1]);
        bytes.extend_from_slice(&[128, 128, 128, 255]);
        bytes.extend_from_slice(&[255, 128, 128, 128]);
        let cancellation = CancellationToken::default();
        let (_, decoded) = decode_gaussian_splat(&bytes, limits, &cancellation)?;
        assert_eq!(decoded.splats().len(), 1);
        cancellation.cancel();
        assert!(matches!(
            decode_gaussian_splat(&bytes, limits, &cancellation),
            Err(GaussianSplatCodecError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn compressed_ksplat_levels_and_spz_versions_are_bounded_and_decodable()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = GaussianSplatCodecLimits::default();
        let cancellation = CancellationToken::default();
        for level in [1_u16, 2] {
            let bytes = compressed_ksplat_fixture(level)?;
            let (format, decoded) = decode_gaussian_splat(&bytes, limits, &cancellation)?;
            assert_eq!(format, NativeFile3DFormat::Ksplat);
            assert_eq!(decoded.splats().len(), 1);
            assert_eq!(decoded.splats()[0].position, [1.0, 2.0, 3.0]);
        }
        for version in [1_u32, 3] {
            let bytes = spz_fixture(version)?;
            let (format, decoded) = decode_gaussian_splat(&bytes, limits, &cancellation)?;
            assert_eq!(format, NativeFile3DFormat::Spz);
            assert_eq!(decoded.splats().len(), 1);
            let rotation = decoded.splats()[0].rotation_wxyz;
            assert!(rotation[0] > 0.99);
            assert!(rotation[1..].iter().all(|value| value.abs() < 0.01));
        }
        Ok(())
    }

    fn compressed_ksplat_fixture(level: u16) -> Result<Vec<u8>, GaussianSplatCodecError> {
        let bytes_per_splat = 6 + 6 + 8 + 4;
        let mut bytes = vec![0_u8; KSPLAT_HEADER_BYTES + KSPLAT_SECTION_HEADER_BYTES];
        bytes[1] = 1;
        put_u32(&mut bytes, 4, 1)?;
        put_u32(&mut bytes, 8, 1)?;
        put_u32(&mut bytes, 12, 1)?;
        put_u32(&mut bytes, 16, 1)?;
        bytes[20..22].copy_from_slice(&level.to_le_bytes());
        let section = KSPLAT_HEADER_BYTES;
        put_u32(&mut bytes, section, 1)?;
        put_u32(&mut bytes, section + 4, 1)?;
        put_u32(&mut bytes, section + 8, 1)?;
        put_u32(&mut bytes, section + 12, 1)?;
        bytes[section + 16..section + 20].copy_from_slice(&1.0_f32.to_le_bytes());
        bytes[section + 20..section + 22].copy_from_slice(&12_u16.to_le_bytes());
        put_u32(&mut bytes, section + 24, 32_767)?;
        put_u32(
            &mut bytes,
            section + 28,
            u32::try_from(bytes_per_splat).map_err(|_| GaussianSplatCodecError::InvalidLength)?,
        )?;
        put_u32(&mut bytes, section + 32, 1)?;
        write_f32s(&mut bytes, &[1.0, 2.0, 3.0]);
        for _ in 0..3 {
            bytes.extend_from_slice(&32_767_u16.to_le_bytes());
        }
        for value in [0.25_f32, 0.5, 1.0] {
            bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
        for value in [1.0_f32, 0.0, 0.0, 0.0] {
            bytes.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&[128, 128, 128, 255]);
        Ok(bytes)
    }

    fn spz_fixture(version: u32) -> Result<Vec<u8>, GaussianSplatCodecError> {
        let mut raw = Vec::new();
        raw.extend_from_slice(&SPZ_MAGIC.to_le_bytes());
        raw.extend_from_slice(&version.to_le_bytes());
        raw.extend_from_slice(&1_u32.to_le_bytes());
        raw.extend_from_slice(&[0, 12, 0, 0]);
        if version == 1 {
            for value in [1.0_f32, 2.0, 3.0] {
                raw.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        } else {
            for value in [4096_i32, 8192, 12_288] {
                raw.extend_from_slice(&value.to_le_bytes()[..3]);
            }
        }
        raw.push(255);
        raw.extend_from_slice(&[128, 128, 128]);
        raw.extend_from_slice(&[128, 128, 128]);
        if version == 3 {
            raw.extend_from_slice(&(3_u32 << 30).to_le_bytes());
        } else {
            raw.extend_from_slice(&[128, 128, 128]);
        }
        let mut encoder = GzBuilder::new()
            .mtime(0)
            .write(Vec::new(), Compression::default());
        encoder
            .write_all(&raw)
            .map_err(|error| GaussianSplatCodecError::Compression(error.to_string()))?;
        encoder
            .finish()
            .map_err(|error| GaussianSplatCodecError::Compression(error.to_string()))
    }
}
