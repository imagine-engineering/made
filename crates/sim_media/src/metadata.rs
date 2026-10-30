use flate2::read::ZlibDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sim_types::normalize_json_non_finite;
use std::{collections::BTreeMap, io::Read, sync::Arc};
use thiserror::Error;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const EXR_SIGNATURE: &[u8; 4] = b"\x76\x2f\x31\x01";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataCarrier {
    Json,
    Png,
    WebP,
    Avif,
    Svg,
    Flac,
    Mp3,
    OggOpus,
    WebM,
    IsobmffVideo,
    Glb,
    Safetensors,
    OpenExr,
    Ply,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MetadataSupport {
    ReadWrite,
    ReadOnly,
    WriteOnly,
    ExplicitNonCarrier,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataLimits {
    pub max_input_bytes: usize,
    pub max_entries: usize,
    pub max_value_bytes: usize,
    pub max_decompressed_bytes: usize,
}

impl Default for MetadataLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 128 * 1024 * 1024,
            max_entries: 4_096,
            max_value_bytes: 16 * 1024 * 1024,
            max_decompressed_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataEntry {
    pub key: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataDiagnostic {
    pub carrier: MetadataCarrier,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetadataDocument {
    carrier: MetadataCarrier,
    support: MetadataSupport,
    original_bytes: Arc<[u8]>,
    entries: Vec<MetadataEntry>,
    diagnostics: Vec<MetadataDiagnostic>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MetadataError {
    #[error("metadata carrier input is {actual} bytes, exceeding the {limit}-byte limit")]
    InputTooLarge { actual: usize, limit: usize },
    #[error("metadata carrier `{0:?}` does not support metadata writes")]
    UnsupportedWrite(MetadataCarrier),
    #[error("metadata write is invalid: {0}")]
    InvalidWrite(String),
    #[error("metadata output exceeds the configured limit")]
    OutputTooLarge,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComfyMetadata {
    pub templates: Vec<String>,
    pub workflow: Option<String>,
    pub prompt: Option<String>,
    pub parameters: Option<String>,
    pub unknown: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetadataWritePolicy {
    pub metadata_enabled: bool,
}

impl Default for MetadataWritePolicy {
    fn default() -> Self {
        Self {
            metadata_enabled: true,
        }
    }
}

impl MetadataCarrier {
    pub fn from_hint(file_name: Option<&str>, mime_type: Option<&str>) -> Self {
        let mime_type = mime_type.unwrap_or_default().to_ascii_lowercase();
        match mime_type.as_str() {
            "application/json" => return Self::Json,
            "image/png" => return Self::Png,
            "image/webp" => return Self::WebP,
            "image/avif" => return Self::Avif,
            "image/svg+xml" => return Self::Svg,
            "audio/flac" | "audio/x-flac" => return Self::Flac,
            "audio/mpeg" => return Self::Mp3,
            "audio/ogg" | "audio/opus" => return Self::OggOpus,
            "video/webm" => return Self::WebM,
            "video/mp4" | "video/quicktime" | "video/x-m4v" => {
                return Self::IsobmffVideo;
            }
            "model/gltf-binary" => return Self::Glb,
            "image/x-exr" => return Self::OpenExr,
            "model/ply" => return Self::Ply,
            _ => {}
        }
        let extension = file_name
            .and_then(|name| name.rsplit_once('.').map(|(_, extension)| extension))
            .unwrap_or_default()
            .to_ascii_lowercase();
        match extension.as_str() {
            "json" => Self::Json,
            "png" => Self::Png,
            "webp" => Self::WebP,
            "avif" => Self::Avif,
            "svg" => Self::Svg,
            "flac" => Self::Flac,
            "mp3" => Self::Mp3,
            "ogg" | "opus" => Self::OggOpus,
            "webm" => Self::WebM,
            "mp4" | "mov" | "m4v" => Self::IsobmffVideo,
            "glb" => Self::Glb,
            "latent" | "safetensors" => Self::Safetensors,
            "exr" => Self::OpenExr,
            "ply" => Self::Ply,
            _ => Self::Unknown,
        }
    }

    fn support(self) -> MetadataSupport {
        match self {
            Self::Png | Self::Svg | Self::Glb | Self::Safetensors => MetadataSupport::ReadWrite,
            Self::WebP
            | Self::Avif
            | Self::Flac
            | Self::Mp3
            | Self::OggOpus
            | Self::WebM
            | Self::IsobmffVideo => MetadataSupport::ReadOnly,
            Self::OpenExr => MetadataSupport::WriteOnly,
            Self::Ply => MetadataSupport::ExplicitNonCarrier,
            Self::Json | Self::Unknown => MetadataSupport::Unsupported,
        }
    }

    fn read_cap(self, limits: &MetadataLimits) -> usize {
        let carrier_cap = match self {
            Self::WebM => 2 * 1024 * 1024,
            Self::IsobmffVideo => 64 * 1024 * 1024,
            Self::Glb => 1024 * 1024,
            Self::Safetensors => 4 * 1024 * 1024 + 8,
            _ => limits.max_input_bytes,
        };
        carrier_cap.min(limits.max_input_bytes)
    }

    fn ignores_disable_metadata(self) -> bool {
        matches!(self, Self::Svg)
    }
}

impl MetadataDocument {
    pub fn parse(
        bytes: impl AsRef<[u8]>,
        file_name: Option<&str>,
        mime_type: Option<&str>,
        limits: MetadataLimits,
    ) -> Result<Self, MetadataError> {
        let bytes = bytes.as_ref();
        if bytes.len() > limits.max_input_bytes {
            return Err(MetadataError::InputTooLarge {
                actual: bytes.len(),
                limit: limits.max_input_bytes,
            });
        }
        let carrier = detect_carrier(bytes, file_name, mime_type);
        let support = carrier.support();
        let cap = carrier.read_cap(&limits);
        let visible = &bytes[..bytes.len().min(cap)];
        let signature_diagnostic = match carrier {
            MetadataCarrier::Mp3
                if !(visible.starts_with(b"ID3")
                    || matches!(visible.get(..2), Some([0xff, second]) if second & 0xe0 == 0xe0)) =>
            {
                Some("invalid MP3 signature; applying the cataloged metadata fallback")
            }
            MetadataCarrier::OggOpus if !visible.starts_with(b"OggS") => {
                Some("invalid Ogg signature; applying the cataloged metadata fallback")
            }
            _ => None,
        };
        let mut parser_diagnostics = Vec::new();
        let parsed = match carrier {
            MetadataCarrier::Png => parse_png(visible, &limits).map(|(entries, diagnostics)| {
                parser_diagnostics = diagnostics;
                entries
            }),
            MetadataCarrier::WebP => parse_webp(visible, &limits),
            MetadataCarrier::Avif => parse_avif(visible, &limits),
            MetadataCarrier::Svg => parse_svg(visible, &limits),
            MetadataCarrier::Flac => parse_flac(visible, &limits),
            MetadataCarrier::Mp3 => parse_mp3(visible, &limits),
            MetadataCarrier::OggOpus => parse_ogg(visible, &limits),
            MetadataCarrier::WebM => parse_webm(visible, &limits),
            MetadataCarrier::IsobmffVideo => parse_isobmff(visible, &limits),
            MetadataCarrier::Glb => parse_glb(visible, &limits),
            MetadataCarrier::Safetensors => parse_safetensors(visible, &limits),
            MetadataCarrier::OpenExr
            | MetadataCarrier::Ply
            | MetadataCarrier::Json
            | MetadataCarrier::Unknown => Ok(Vec::new()),
        };
        let mut diagnostics = signature_diagnostic
            .into_iter()
            .map(|message| MetadataDiagnostic {
                carrier,
                message: message.to_owned(),
            })
            .collect::<Vec<_>>();
        diagnostics.extend(
            parser_diagnostics
                .into_iter()
                .map(|message| MetadataDiagnostic { carrier, message }),
        );
        let entries = match parsed {
            Ok(entries) => entries,
            Err(message) => {
                diagnostics.push(MetadataDiagnostic { carrier, message });
                Vec::new()
            }
        };
        if bytes.len() > cap {
            diagnostics.push(MetadataDiagnostic {
                carrier,
                message: format!(
                    "only the first {cap} bytes are part of the cataloged metadata scan"
                ),
            });
        }
        Ok(Self {
            carrier,
            support,
            original_bytes: Arc::from(bytes),
            entries,
            diagnostics,
        })
    }

    pub fn carrier(&self) -> MetadataCarrier {
        self.carrier
    }

    pub fn support(&self) -> MetadataSupport {
        self.support
    }

    pub fn original_bytes(&self) -> &[u8] {
        &self.original_bytes
    }

    pub fn entries(&self) -> &[MetadataEntry] {
        &self.entries
    }

    pub fn diagnostics(&self) -> &[MetadataDiagnostic] {
        &self.diagnostics
    }

    pub fn get_case_insensitive(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.key.eq_ignore_ascii_case(key))
            .map(|entry| entry.value.as_str())
    }

    pub fn sim_metadata(&self) -> ComfyMetadata {
        let mut metadata = ComfyMetadata::default();
        for entry in &self.entries {
            match entry.key.to_ascii_lowercase().as_str() {
                "templates" => metadata.templates.push(entry.value.clone()),
                "workflow" => metadata.workflow = Some(entry.value.clone()),
                "prompt" => metadata.prompt = Some(entry.value.clone()),
                "parameters" => metadata.parameters = Some(entry.value.clone()),
                _ => {
                    metadata
                        .unknown
                        .insert(entry.key.clone(), entry.value.clone());
                }
            }
        }
        metadata
    }

    pub fn embed_sim_metadata(
        &self,
        fields: &BTreeMap<String, String>,
        policy: MetadataWritePolicy,
        limits: &MetadataLimits,
    ) -> Result<Vec<u8>, MetadataError> {
        if !policy.metadata_enabled && !self.carrier.ignores_disable_metadata() {
            return Ok(self.original_bytes.to_vec());
        }
        validate_write_fields(fields, limits)?;
        let output = match self.carrier {
            MetadataCarrier::Png => write_png(&self.original_bytes, fields)?,
            MetadataCarrier::Svg => write_svg(&self.original_bytes, fields)?,
            MetadataCarrier::Glb => write_glb(&self.original_bytes, fields)?,
            MetadataCarrier::Safetensors => {
                write_safetensors(&self.original_bytes, fields, limits)?
            }
            MetadataCarrier::OpenExr => write_open_exr(&self.original_bytes, fields)?,
            carrier => return Err(MetadataError::UnsupportedWrite(carrier)),
        };
        if output.len() > limits.max_input_bytes {
            return Err(MetadataError::OutputTooLarge);
        }
        Ok(output)
    }
}

pub fn detect_carrier(
    bytes: &[u8],
    file_name: Option<&str>,
    mime_type: Option<&str>,
) -> MetadataCarrier {
    if bytes.starts_with(PNG_SIGNATURE) {
        return MetadataCarrier::Png;
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return MetadataCarrier::WebP;
    }
    if bytes.starts_with(b"fLaC") {
        return MetadataCarrier::Flac;
    }
    if bytes.starts_with(b"OggS") {
        return MetadataCarrier::OggOpus;
    }
    if bytes.starts_with(b"ID3")
        || matches!(bytes.get(..2), Some([0xff, second]) if second & 0xe0 == 0xe0)
    {
        return MetadataCarrier::Mp3;
    }
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return MetadataCarrier::WebM;
    }
    if bytes.starts_with(b"glTF") {
        return MetadataCarrier::Glb;
    }
    if bytes.starts_with(EXR_SIGNATURE) {
        return MetadataCarrier::OpenExr;
    }
    if bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r\n") {
        return MetadataCarrier::Ply;
    }
    if bytes.get(4..8) == Some(b"ftyp") {
        if bytes.get(8..12) == Some(b"avif") || bytes.get(8..12) == Some(b"avis") {
            return MetadataCarrier::Avif;
        }
        return MetadataCarrier::IsobmffVideo;
    }
    let first_non_whitespace = bytes
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace());
    if matches!(first_non_whitespace, Some(b'{') | Some(b'[')) {
        return MetadataCarrier::Json;
    }
    if bytes
        .windows(4)
        .take(512)
        .any(|window| window.eq_ignore_ascii_case(b"<svg"))
    {
        return MetadataCarrier::Svg;
    }
    MetadataCarrier::from_hint(file_name, mime_type)
}

fn validate_write_fields(
    fields: &BTreeMap<String, String>,
    limits: &MetadataLimits,
) -> Result<(), MetadataError> {
    if fields.len() > limits.max_entries {
        return Err(MetadataError::InvalidWrite(
            "metadata field count exceeds the configured limit".to_owned(),
        ));
    }
    for (key, value) in fields {
        if key.is_empty() || key.as_bytes().contains(&0) {
            return Err(MetadataError::InvalidWrite(
                "metadata keys must be nonempty and NUL-free".to_owned(),
            ));
        }
        if value.len() > limits.max_value_bytes {
            return Err(MetadataError::InvalidWrite(format!(
                "metadata value `{key}` exceeds the configured limit"
            )));
        }
    }
    Ok(())
}

fn push_entry(
    entries: &mut Vec<MetadataEntry>,
    key: String,
    value: String,
    limits: &MetadataLimits,
) -> Result<(), String> {
    if entries.len() >= limits.max_entries {
        return Err("metadata entry count exceeds the configured limit".to_owned());
    }
    if value.len() > limits.max_value_bytes {
        return Err(format!("metadata value `{key}` exceeds its byte limit"));
    }
    entries.push(MetadataEntry { key, value });
    Ok(())
}

fn read_be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let array: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_be_bytes(array))
}

fn read_le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let array: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(array))
}

fn read_le_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let array: [u8; 8] = bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?;
    Some(u64::from_le_bytes(array))
}

fn decode_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches('\0')
        .to_owned()
}

struct PngChunk<'a> {
    chunk_type: &'a [u8],
    data: &'a [u8],
    end: usize,
}

fn read_png_chunk(bytes: &[u8], offset: usize) -> Result<PngChunk<'_>, String> {
    let length = usize::try_from(
        read_be_u32(bytes, offset).ok_or_else(|| "truncated PNG chunk length".to_owned())?,
    )
    .map_err(|_| "PNG chunk length does not fit this platform".to_owned())?;
    let chunk_type_start = offset
        .checked_add(4)
        .ok_or_else(|| "PNG chunk offset overflow".to_owned())?;
    let data_start = offset
        .checked_add(8)
        .ok_or_else(|| "PNG chunk offset overflow".to_owned())?;
    let data_end = data_start
        .checked_add(length)
        .ok_or_else(|| "PNG chunk length overflow".to_owned())?;
    let chunk_end = data_end
        .checked_add(4)
        .ok_or_else(|| "PNG chunk length overflow".to_owned())?;
    let chunk_type = bytes
        .get(chunk_type_start..data_start)
        .ok_or_else(|| "truncated PNG chunk type".to_owned())?;
    let data = bytes
        .get(data_start..data_end)
        .ok_or_else(|| "truncated PNG chunk payload".to_owned())?;
    if chunk_end > bytes.len() {
        return Err("truncated PNG chunk checksum".to_owned());
    }
    Ok(PngChunk {
        chunk_type,
        data,
        end: chunk_end,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PngSourceTransparency {
    Grayscale(u16),
    Rgb([u16; 3]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PngSourcePixelFormat {
    pub bit_depth: u8,
    pub color_type: u8,
    pub transparency: Option<PngSourceTransparency>,
}

#[derive(Debug)]
pub(crate) enum PngSourcePixelError {
    Malformed(String),
    Cancelled,
}

pub(crate) fn png_source_pixel_format(
    bytes: &[u8],
    cancellation: &sim_tensor::CancellationToken,
) -> Result<PngSourcePixelFormat, PngSourcePixelError> {
    let malformed = PngSourcePixelError::Malformed;
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Err(malformed("invalid PNG signature".to_owned()));
    }
    cancellation
        .check()
        .map_err(|_| PngSourcePixelError::Cancelled)?;
    let header = read_png_chunk(bytes, PNG_SIGNATURE.len()).map_err(malformed)?;
    let [_, _, _, _, _, _, _, _, bit_depth, color_type, _, _, _] = header.data else {
        return Err(malformed("invalid PNG IHDR payload".to_owned()));
    };
    if header.chunk_type != b"IHDR" {
        return Err(malformed("PNG first chunk is not IHDR".to_owned()));
    }
    let mut format = PngSourcePixelFormat {
        bit_depth: *bit_depth,
        color_type: *color_type,
        transparency: None,
    };
    let mut offset = header.end;
    while offset < bytes.len() {
        cancellation
            .check()
            .map_err(|_| PngSourcePixelError::Cancelled)?;
        let chunk = read_png_chunk(bytes, offset).map_err(malformed)?;
        if chunk.chunk_type == b"IDAT" {
            return Ok(format);
        }
        if chunk.chunk_type == b"IEND" {
            break;
        }
        if chunk.chunk_type == b"tRNS" {
            format.transparency = match (format.color_type, chunk.data) {
                (0, [high, low]) => Some(PngSourceTransparency::Grayscale(u16::from_be_bytes([
                    *high, *low,
                ]))),
                (
                    2,
                    [
                        red_high,
                        red_low,
                        green_high,
                        green_low,
                        blue_high,
                        blue_low,
                    ],
                ) => Some(PngSourceTransparency::Rgb([
                    u16::from_be_bytes([*red_high, *red_low]),
                    u16::from_be_bytes([*green_high, *green_low]),
                    u16::from_be_bytes([*blue_high, *blue_low]),
                ])),
                (0 | 2, _) => return Err(malformed("invalid PNG tRNS payload".to_owned())),
                _ => None,
            };
        }
        offset = chunk.end;
    }
    Err(malformed("PNG has no image data chunk".to_owned()))
}

fn parse_png(
    bytes: &[u8],
    limits: &MetadataLimits,
) -> Result<(Vec<MetadataEntry>, Vec<String>), String> {
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Err("invalid PNG signature".to_owned());
    }
    let mut offset = PNG_SIGNATURE.len();
    let mut entries = Vec::new();
    let mut diagnostics = Vec::new();
    while offset < bytes.len() {
        let PngChunk {
            chunk_type,
            data,
            end: chunk_end,
        } = read_png_chunk(bytes, offset)?;
        if matches!(chunk_type, b"tEXt" | b"comf") {
            if let Some(separator) = data.iter().position(|byte| *byte == 0) {
                let key = decode_text(&data[..separator]);
                let value = decode_text(&data[separator + 1..]);
                push_entry(&mut entries, key, value, limits)?;
            } else {
                diagnostics.push(format!(
                    "{} chunk has no keyword separator",
                    decode_text(chunk_type)
                ));
            }
        } else if chunk_type == b"iTXt" {
            if let Err(message) = parse_itxt(data, &mut entries, limits) {
                diagnostics.push(message);
            }
        }
        offset = chunk_end;
        if chunk_type == b"IEND" {
            return Ok((entries, diagnostics));
        }
    }
    Err("PNG has no complete IEND chunk".to_owned())
}

fn parse_itxt(
    data: &[u8],
    entries: &mut Vec<MetadataEntry>,
    limits: &MetadataLimits,
) -> Result<(), String> {
    let keyword_end = data
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| "iTXt keyword is unterminated".to_owned())?;
    let key = decode_text(&data[..keyword_end]);
    let mut position = keyword_end + 1;
    let compression_flag = *data
        .get(position)
        .ok_or_else(|| "iTXt compression flag is missing".to_owned())?;
    position += 1;
    let compression_method = *data
        .get(position)
        .ok_or_else(|| "iTXt compression method is missing".to_owned())?;
    position += 1;
    for _ in 0..2 {
        let relative_end = data
            .get(position..)
            .and_then(|tail| tail.iter().position(|byte| *byte == 0))
            .ok_or_else(|| "iTXt language field is unterminated".to_owned())?;
        position = position
            .checked_add(relative_end + 1)
            .ok_or_else(|| "iTXt field offset overflow".to_owned())?;
    }
    let payload = data
        .get(position..)
        .ok_or_else(|| "iTXt payload is missing".to_owned())?;
    let value = if compression_flag == 0 {
        decode_text(payload)
    } else if compression_flag == 1 && compression_method == 0 {
        let limit = u64::try_from(limits.max_decompressed_bytes)
            .map_err(|_| "decompression limit is unsupported".to_owned())?;
        let decoder = ZlibDecoder::new(payload);
        let mut output = Vec::new();
        decoder
            .take(limit.saturating_add(1))
            .read_to_end(&mut output)
            .map_err(|error| format!("iTXt decompression failed: {error}"))?;
        if output.len() > limits.max_decompressed_bytes {
            return Err("iTXt decompressed value exceeds its limit".to_owned());
        }
        decode_text(&output)
    } else {
        return Ok(());
    };
    push_entry(entries, key, value, limits)
}

fn parse_webp(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if !bytes.starts_with(b"RIFF") || bytes.get(8..12) != Some(b"WEBP") {
        return Err("invalid RIFF/WEBP signature".to_owned());
    }
    let mut offset = 12usize;
    let mut entries = Vec::new();
    while offset < bytes.len() {
        let chunk_type = bytes
            .get(offset..offset.saturating_add(4))
            .ok_or_else(|| "truncated WebP chunk type".to_owned())?;
        let length = usize::try_from(
            read_le_u32(bytes, offset.saturating_add(4))
                .ok_or_else(|| "truncated WebP chunk length".to_owned())?,
        )
        .map_err(|_| "WebP chunk length does not fit this platform".to_owned())?;
        let data_start = offset
            .checked_add(8)
            .ok_or_else(|| "WebP chunk offset overflow".to_owned())?;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| "WebP chunk length overflow".to_owned())?;
        let payload = bytes
            .get(data_start..data_end)
            .ok_or_else(|| "truncated WebP chunk payload".to_owned())?;
        if chunk_type == b"EXIF" {
            entries.extend(parse_exif(payload, limits)?);
        }
        offset = data_end
            .checked_add(length % 2)
            .ok_or_else(|| "WebP padding overflow".to_owned())?;
    }
    Ok(entries)
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

fn read_tiff_u16(bytes: &[u8], offset: usize, byte_order: ByteOrder) -> Option<u16> {
    let array: [u8; 2] = bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
    Some(match byte_order {
        ByteOrder::Little => u16::from_le_bytes(array),
        ByteOrder::Big => u16::from_be_bytes(array),
    })
}

fn read_tiff_u32(bytes: &[u8], offset: usize, byte_order: ByteOrder) -> Option<u32> {
    let array: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(match byte_order {
        ByteOrder::Little => u32::from_le_bytes(array),
        ByteOrder::Big => u32::from_be_bytes(array),
    })
}

fn parse_exif(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    let bytes = bytes.strip_prefix(b"Exif\0\0").unwrap_or(bytes);
    let byte_order = match bytes.get(..2) {
        Some(b"II") => ByteOrder::Little,
        Some(b"MM") => ByteOrder::Big,
        _ => return parse_prefixed_text(bytes, limits),
    };
    if read_tiff_u16(bytes, 2, byte_order) != Some(42) {
        return Err("invalid TIFF header in EXIF metadata".to_owned());
    }
    let ifd_offset = usize::try_from(
        read_tiff_u32(bytes, 4, byte_order).ok_or_else(|| "missing TIFF IFD offset".to_owned())?,
    )
    .map_err(|_| "TIFF IFD offset does not fit this platform".to_owned())?;
    let count = usize::from(
        read_tiff_u16(bytes, ifd_offset, byte_order)
            .ok_or_else(|| "truncated TIFF IFD".to_owned())?,
    );
    let mut entries = Vec::new();
    for index in 0..count.min(limits.max_entries) {
        let entry_offset = ifd_offset
            .checked_add(2)
            .and_then(|offset| offset.checked_add(index.checked_mul(12)?))
            .ok_or_else(|| "TIFF IFD offset overflow".to_owned())?;
        let tag = read_tiff_u16(bytes, entry_offset, byte_order)
            .ok_or_else(|| "truncated TIFF tag".to_owned())?;
        let field_type = read_tiff_u16(bytes, entry_offset + 2, byte_order)
            .ok_or_else(|| "truncated TIFF field type".to_owned())?;
        let count = usize::try_from(
            read_tiff_u32(bytes, entry_offset + 4, byte_order)
                .ok_or_else(|| "truncated TIFF field count".to_owned())?,
        )
        .map_err(|_| "TIFF field count does not fit this platform".to_owned())?;
        if !matches!(field_type, 1 | 2 | 7) || count == 0 || count > limits.max_value_bytes {
            continue;
        }
        let value = if count <= 4 {
            bytes
                .get(entry_offset + 8..entry_offset + 8 + count)
                .ok_or_else(|| "truncated inline TIFF value".to_owned())?
        } else {
            let value_offset = usize::try_from(
                read_tiff_u32(bytes, entry_offset + 8, byte_order)
                    .ok_or_else(|| "truncated TIFF value offset".to_owned())?,
            )
            .map_err(|_| "TIFF value offset does not fit this platform".to_owned())?;
            bytes
                .get(value_offset..value_offset.saturating_add(count))
                .ok_or_else(|| "TIFF value is outside EXIF payload".to_owned())?
        };
        let mut parsed = parse_prefixed_text(value, limits)?;
        if parsed.is_empty() && tag == 0x9286 {
            let json_start = value.iter().position(|byte| *byte == b'{');
            if let Some(json_start) = json_start {
                let json = value.get(json_start..).unwrap_or_default();
                let json_end = json
                    .iter()
                    .rposition(|byte| *byte != 0 && !byte.is_ascii_whitespace())
                    .map_or(0, |index| index + 1);
                parsed = json_object_entries(&json[..json_end], limits)?;
            }
        }
        entries.extend(parsed);
    }
    Ok(entries)
}

fn parse_prefixed_text(
    bytes: &[u8],
    limits: &MetadataLimits,
) -> Result<Vec<MetadataEntry>, String> {
    let text = decode_text(bytes);
    let mut entries = Vec::new();
    for candidate in text.split(['\0', '\n', '\r']) {
        let candidate = candidate.trim();
        let split = candidate
            .split_once(':')
            .or_else(|| candidate.split_once('='));
        if let Some((key, value)) = split {
            if matches!(
                key.trim().to_ascii_lowercase().as_str(),
                "prompt" | "workflow" | "templates" | "parameters"
            ) {
                push_entry(
                    &mut entries,
                    key.trim().to_owned(),
                    value.trim().to_owned(),
                    limits,
                )?;
            }
        }
    }
    Ok(entries)
}

fn parse_avif(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if bytes.get(4..8) != Some(b"ftyp")
        || !matches!(bytes.get(8..12), Some(b"avif") | Some(b"avis"))
    {
        return Err("invalid AVIF signature".to_owned());
    }
    let meta = find_box(bytes, 0, bytes.len(), b"meta")?
        .ok_or_else(|| "AVIF has no meta box".to_owned())?;
    let children_start = meta
        .0
        .checked_add(4)
        .ok_or_else(|| "AVIF meta offset overflow".to_owned())?;
    let iinf = find_box(bytes, children_start, meta.1, b"iinf")?
        .ok_or_else(|| "AVIF has no iinf box".to_owned())?;
    let exif_item =
        parse_avif_iinf(bytes, iinf)?.ok_or_else(|| "AVIF has no EXIF item".to_owned())?;
    let iloc = find_box(bytes, children_start, meta.1, b"iloc")?
        .ok_or_else(|| "AVIF has no iloc box".to_owned())?;
    let (offset, length) = parse_avif_iloc(bytes, iloc, exif_item)?
        .ok_or_else(|| "AVIF EXIF item has no extent".to_owned())?;
    let payload = bytes
        .get(offset..offset.saturating_add(length))
        .ok_or_else(|| "AVIF EXIF extent is outside the file".to_owned())?;
    let tiff_offset = if payload.len() >= 4 {
        usize::try_from(read_be_u32(payload, 0).unwrap_or(0))
            .map_err(|_| "AVIF TIFF offset does not fit this platform".to_owned())?
            .saturating_add(4)
    } else {
        0
    };
    parse_exif(payload.get(tiff_offset..).unwrap_or(payload), limits)
}

fn find_box(
    bytes: &[u8],
    start: usize,
    end: usize,
    box_type: &[u8; 4],
) -> Result<Option<(usize, usize)>, String> {
    let mut position = start;
    while position.saturating_add(8) <= end {
        let size = usize::try_from(
            read_be_u32(bytes, position).ok_or_else(|| "truncated box size".to_owned())?,
        )
        .map_err(|_| "box size does not fit this platform".to_owned())?;
        if size < 8 {
            return Err("box size is smaller than its header".to_owned());
        }
        let box_end = position
            .checked_add(size)
            .ok_or_else(|| "box size overflow".to_owned())?;
        if box_end > end || box_end > bytes.len() {
            return Err("box extends outside its parent".to_owned());
        }
        if bytes.get(position + 4..position + 8) == Some(box_type) {
            return Ok(Some((position + 8, box_end)));
        }
        position = box_end;
    }
    Ok(None)
}

fn parse_avif_iinf(bytes: &[u8], range: (usize, usize)) -> Result<Option<u32>, String> {
    let version = *bytes
        .get(range.0)
        .ok_or_else(|| "truncated iinf version".to_owned())?;
    let mut position = range.0.saturating_add(4);
    let entry_count = if version == 0 {
        let count = read_tiff_u16(bytes, position, ByteOrder::Big)
            .ok_or_else(|| "truncated iinf count".to_owned())?;
        position += 2;
        u32::from(count)
    } else {
        let count =
            read_be_u32(bytes, position).ok_or_else(|| "truncated iinf count".to_owned())?;
        position += 4;
        count
    };
    for _ in 0..entry_count {
        let size = usize::try_from(
            read_be_u32(bytes, position).ok_or_else(|| "truncated infe size".to_owned())?,
        )
        .map_err(|_| "infe size does not fit this platform".to_owned())?;
        if size < 16 || position.saturating_add(size) > range.1 {
            return Err("invalid infe range".to_owned());
        }
        if bytes.get(position + 4..position + 8) == Some(b"infe") {
            let infe_version = *bytes
                .get(position + 8)
                .ok_or_else(|| "truncated infe version".to_owned())?;
            if infe_version >= 2 {
                let id_offset = position + 12;
                let (item_id, type_offset) = if infe_version == 2 {
                    (
                        u32::from(
                            read_tiff_u16(bytes, id_offset, ByteOrder::Big)
                                .ok_or_else(|| "truncated infe item ID".to_owned())?,
                        ),
                        id_offset + 4,
                    )
                } else {
                    (
                        read_be_u32(bytes, id_offset)
                            .ok_or_else(|| "truncated infe item ID".to_owned())?,
                        id_offset + 6,
                    )
                };
                if bytes.get(type_offset..type_offset + 4) == Some(b"Exif") {
                    return Ok(Some(item_id));
                }
            }
        }
        position += size;
    }
    Ok(None)
}

fn read_sized_uint(bytes: &[u8], position: &mut usize, size: usize) -> Option<u64> {
    if size > 8 {
        return None;
    }
    let mut value = 0u64;
    for byte in bytes.get(*position..position.checked_add(size)?)? {
        value = value.checked_shl(8)?.checked_add(u64::from(*byte))?;
    }
    *position = position.checked_add(size)?;
    Some(value)
}

fn parse_avif_iloc(
    bytes: &[u8],
    range: (usize, usize),
    wanted_item_id: u32,
) -> Result<Option<(usize, usize)>, String> {
    let version = *bytes
        .get(range.0)
        .ok_or_else(|| "truncated iloc version".to_owned())?;
    let mut position = range.0.saturating_add(4);
    let first_sizes = *bytes
        .get(position)
        .ok_or_else(|| "truncated iloc sizes".to_owned())?;
    position += 1;
    let second_sizes = *bytes
        .get(position)
        .ok_or_else(|| "truncated iloc sizes".to_owned())?;
    position += 1;
    let offset_size = usize::from(first_sizes >> 4);
    let length_size = usize::from(first_sizes & 0x0f);
    let base_offset_size = usize::from(second_sizes >> 4);
    let index_size = if matches!(version, 1 | 2) {
        usize::from(second_sizes & 0x0f)
    } else {
        0
    };
    let item_count = if version < 2 {
        let count = read_tiff_u16(bytes, position, ByteOrder::Big)
            .ok_or_else(|| "truncated iloc item count".to_owned())?;
        position += 2;
        u32::from(count)
    } else {
        let count =
            read_be_u32(bytes, position).ok_or_else(|| "truncated iloc item count".to_owned())?;
        position += 4;
        count
    };
    for _ in 0..item_count {
        let item_id = if version < 2 {
            let id = read_tiff_u16(bytes, position, ByteOrder::Big)
                .ok_or_else(|| "truncated iloc item ID".to_owned())?;
            position += 2;
            u32::from(id)
        } else {
            let id =
                read_be_u32(bytes, position).ok_or_else(|| "truncated iloc item ID".to_owned())?;
            position += 4;
            id
        };
        if matches!(version, 1 | 2) {
            position = position.saturating_add(2);
        }
        position = position.saturating_add(2);
        let base_offset = read_sized_uint(bytes, &mut position, base_offset_size)
            .ok_or_else(|| "truncated iloc base offset".to_owned())?;
        let extent_count = read_tiff_u16(bytes, position, ByteOrder::Big)
            .ok_or_else(|| "truncated iloc extent count".to_owned())?;
        position += 2;
        for _ in 0..extent_count {
            if index_size > 0 {
                read_sized_uint(bytes, &mut position, index_size)
                    .ok_or_else(|| "truncated iloc extent index".to_owned())?;
            }
            let extent_offset = read_sized_uint(bytes, &mut position, offset_size)
                .ok_or_else(|| "truncated iloc extent offset".to_owned())?;
            let extent_length = read_sized_uint(bytes, &mut position, length_size)
                .ok_or_else(|| "truncated iloc extent length".to_owned())?;
            if item_id == wanted_item_id {
                let absolute = base_offset
                    .checked_add(extent_offset)
                    .ok_or_else(|| "AVIF extent offset overflow".to_owned())?;
                return Ok(Some((
                    usize::try_from(absolute)
                        .map_err(|_| "AVIF extent offset does not fit this platform".to_owned())?,
                    usize::try_from(extent_length)
                        .map_err(|_| "AVIF extent length does not fit this platform".to_owned())?,
                )));
            }
        }
        if position > range.1 {
            return Err("iloc item extends outside its box".to_owned());
        }
    }
    Ok(None)
}

fn parse_svg(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("SVG is not UTF-8: {error}"))?;
    let lower = text.to_ascii_lowercase();
    let metadata_start = lower
        .find("<metadata")
        .ok_or_else(|| "SVG has no metadata element".to_owned())?;
    let cdata_relative = lower[metadata_start..]
        .find("<![cdata[")
        .ok_or_else(|| "SVG metadata has no CDATA payload".to_owned())?;
    let cdata_start = metadata_start + cdata_relative + "<![cdata[".len();
    let cdata_end = text[cdata_start..]
        .find("]]>")
        .map(|offset| cdata_start + offset)
        .ok_or_else(|| "SVG metadata CDATA is unterminated".to_owned())?;
    json_object_entries(&text.as_bytes()[cdata_start..cdata_end], limits)
}

fn json_object_entries(
    bytes: &[u8],
    limits: &MetadataLimits,
) -> Result<Vec<MetadataEntry>, String> {
    let (normalized, _) = normalize_json_non_finite(bytes);
    let value: Value = serde_json::from_slice(&normalized)
        .map_err(|error| format!("metadata JSON is invalid: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "metadata JSON root is not an object".to_owned())?;
    let mut entries = Vec::new();
    for (key, value) in object {
        let value = match value {
            Value::String(value) => value.clone(),
            value => serde_json::to_string(value)
                .map_err(|error| format!("metadata JSON value is invalid: {error}"))?,
        };
        push_entry(&mut entries, key.clone(), value, limits)?;
    }
    Ok(entries)
}

fn parse_flac(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if !bytes.starts_with(b"fLaC") {
        return Err("invalid FLAC signature".to_owned());
    }
    let mut position = 4usize;
    while position < bytes.len() {
        let header = *bytes
            .get(position)
            .ok_or_else(|| "truncated FLAC block header".to_owned())?;
        let length_bytes = bytes
            .get(position + 1..position + 4)
            .ok_or_else(|| "truncated FLAC block length".to_owned())?;
        let length = (usize::from(length_bytes[0]) << 16)
            | (usize::from(length_bytes[1]) << 8)
            | usize::from(length_bytes[2]);
        let data_start = position + 4;
        let data_end = data_start
            .checked_add(length)
            .ok_or_else(|| "FLAC block length overflow".to_owned())?;
        let payload = bytes
            .get(data_start..data_end)
            .ok_or_else(|| "truncated FLAC metadata block".to_owned())?;
        if header & 0x7f == 4 {
            return parse_vorbis_comments(payload, limits);
        }
        position = data_end;
        if header & 0x80 != 0 {
            break;
        }
    }
    Ok(Vec::new())
}

fn parse_vorbis_comments(
    bytes: &[u8],
    limits: &MetadataLimits,
) -> Result<Vec<MetadataEntry>, String> {
    let vendor_length = usize::try_from(
        read_le_u32(bytes, 0).ok_or_else(|| "truncated Vorbis vendor length".to_owned())?,
    )
    .map_err(|_| "Vorbis vendor length does not fit this platform".to_owned())?;
    let mut position = 4usize
        .checked_add(vendor_length)
        .ok_or_else(|| "Vorbis vendor length overflow".to_owned())?;
    if position > bytes.len() {
        return Err("Vorbis vendor is truncated".to_owned());
    }
    let count = usize::try_from(
        read_le_u32(bytes, position).ok_or_else(|| "truncated Vorbis comment count".to_owned())?,
    )
    .map_err(|_| "Vorbis comment count does not fit this platform".to_owned())?;
    position += 4;
    if count > limits.max_entries {
        return Err("Vorbis comment count exceeds the configured limit".to_owned());
    }
    let mut entries = Vec::new();
    for _ in 0..count {
        let length = usize::try_from(
            read_le_u32(bytes, position)
                .ok_or_else(|| "truncated Vorbis comment length".to_owned())?,
        )
        .map_err(|_| "Vorbis comment length does not fit this platform".to_owned())?;
        position += 4;
        let comment = bytes
            .get(position..position.saturating_add(length))
            .ok_or_else(|| "truncated Vorbis comment".to_owned())?;
        position += length;
        let comment = decode_text(comment);
        if let Some((key, value)) = comment.split_once('=') {
            push_entry(&mut entries, key.to_owned(), value.to_owned(), limits)?;
        }
    }
    Ok(entries)
}

fn parse_mp3(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    let scan_end = bytes
        .windows(2)
        .position(|window| window[0] == 0xff && window[1] & 0xe0 == 0xe0)
        .unwrap_or(bytes.len());
    parse_named_json_markers(&bytes[..scan_end], limits)
}

fn parse_named_json_markers(
    bytes: &[u8],
    limits: &MetadataLimits,
) -> Result<Vec<MetadataEntry>, String> {
    let mut entries = Vec::new();
    for key in ["prompt", "workflow", "templates", "parameters"] {
        let mut start = 0usize;
        while let Some(relative) = bytes.get(start..).and_then(|tail| {
            tail.windows(key.len() + 1).position(|window| {
                window[..key.len()].eq_ignore_ascii_case(key.as_bytes())
                    && matches!(window[key.len()], 0 | b'=')
            })
        }) {
            let value_start = start + relative + key.len() + 1;
            let tail = bytes.get(value_start..).unwrap_or_default();
            let value_end = if key == "parameters" {
                tail.iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(tail.len())
            } else {
                balanced_json_length(tail).unwrap_or_else(|| {
                    tail.iter()
                        .position(|byte| *byte == 0)
                        .unwrap_or(tail.len())
                })
            };
            if value_end > 0 {
                push_entry(
                    &mut entries,
                    key.to_owned(),
                    decode_text(&tail[..value_end]),
                    limits,
                )?;
            }
            start = value_start.saturating_add(value_end);
            if start >= bytes.len() {
                break;
            }
        }
    }
    Ok(entries)
}

fn balanced_json_length(bytes: &[u8]) -> Option<usize> {
    let start = bytes.iter().position(|byte| matches!(byte, b'{' | b'['))?;
    let opening = bytes[start];
    let closing = if opening == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
        } else if byte == opening {
            depth = depth.checked_add(1)?;
        } else if byte == closing {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return index.checked_add(1);
            }
        }
    }
    None
}

fn parse_ogg(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if !bytes.starts_with(b"OggS") {
        return parse_named_json_markers(bytes, limits);
    }
    let mut position = 0usize;
    let mut packet = Vec::new();
    while position < bytes.len() {
        if bytes.get(position..position + 4) != Some(b"OggS") {
            return Err("invalid Ogg page signature".to_owned());
        }
        let segment_count = usize::from(
            *bytes
                .get(position + 26)
                .ok_or_else(|| "truncated Ogg page header".to_owned())?,
        );
        let table = bytes
            .get(position + 27..position + 27 + segment_count)
            .ok_or_else(|| "truncated Ogg lacing table".to_owned())?;
        let data_length = table.iter().map(|value| usize::from(*value)).sum::<usize>();
        let mut data_position = position + 27 + segment_count;
        let page_end = data_position
            .checked_add(data_length)
            .ok_or_else(|| "Ogg page length overflow".to_owned())?;
        if page_end > bytes.len() {
            return Err("truncated Ogg page payload".to_owned());
        }
        for segment in table {
            let segment_length = usize::from(*segment);
            packet.extend_from_slice(
                bytes
                    .get(data_position..data_position + segment_length)
                    .ok_or_else(|| "truncated Ogg segment".to_owned())?,
            );
            if packet.len() > limits.max_decompressed_bytes {
                return Err("Ogg metadata packet exceeds its limit".to_owned());
            }
            data_position += segment_length;
            if *segment < 255 {
                if let Some(payload) = packet.strip_prefix(b"OpusTags") {
                    return parse_vorbis_comments(payload, limits);
                }
                if packet.get(1..7) == Some(b"vorbis") && packet.first() == Some(&3) {
                    return parse_vorbis_comments(&packet[7..], limits);
                }
                packet.clear();
            }
        }
        position = page_end;
    }
    Ok(Vec::new())
}

fn read_ebml_vint(bytes: &[u8], position: usize) -> Option<(usize, usize)> {
    let first = *bytes.get(position)?;
    let length = usize::try_from(first.leading_zeros())
        .ok()?
        .checked_add(1)?;
    if length > 8 || position.checked_add(length)? > bytes.len() {
        return None;
    }
    let mut value = usize::from(first & (0xff >> length));
    for byte in bytes.get(position + 1..position + length)? {
        value = value.checked_shl(8)?.checked_add(usize::from(*byte))?;
    }
    let unknown_length = 1usize.checked_shl(u32::try_from(7usize.checked_mul(length)?).ok()?)? - 1;
    if value == 0 || value == unknown_length {
        return None;
    }
    Some((value, length))
}

fn parse_webm(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if !bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Err("invalid WebM EBML signature".to_owned());
    }
    let mut entries = Vec::new();
    let simple_tag = [0x67, 0xc8];
    let mut position = 0usize;
    while position + simple_tag.len() < bytes.len() {
        let Some(relative) = bytes[position..]
            .windows(simple_tag.len())
            .position(|window| window == simple_tag)
        else {
            break;
        };
        let tag_position = position + relative;
        let size_position = tag_position + simple_tag.len();
        let Some((length, length_bytes)) = read_ebml_vint(bytes, size_position) else {
            position = size_position;
            continue;
        };
        let payload_start = size_position + length_bytes;
        let payload_end = payload_start.saturating_add(length).min(bytes.len());
        if let (Some(name), Some(value)) = (
            find_ebml_text(bytes, payload_start, payload_end, &[0x45, 0xa3]),
            find_ebml_text(bytes, payload_start, payload_end, &[0x44, 0x87]),
        ) {
            push_entry(&mut entries, name, value, limits)?;
        }
        position = payload_end.max(size_position + 1);
    }
    Ok(entries)
}

fn find_ebml_text(bytes: &[u8], start: usize, end: usize, identifier: &[u8]) -> Option<String> {
    let relative = bytes
        .get(start..end)?
        .windows(identifier.len())
        .position(|window| window == identifier)?;
    let size_position = start.checked_add(relative)?.checked_add(identifier.len())?;
    let (length, length_bytes) = read_ebml_vint(bytes, size_position)?;
    let value_start = size_position.checked_add(length_bytes)?;
    let value_end = value_start.checked_add(length)?;
    if value_end > end {
        return None;
    }
    Some(
        decode_text(bytes.get(value_start..value_end)?)
            .trim()
            .to_owned(),
    )
}

fn parse_isobmff(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if bytes.get(4..8) != Some(b"ftyp") {
        return Err("invalid ISOBMFF signature".to_owned());
    }
    let user_data = if let Some(user_data) = find_box(bytes, 0, bytes.len(), b"udta")? {
        Some(user_data)
    } else if let Some(movie) = find_box(bytes, 0, bytes.len(), b"moov")? {
        find_box(bytes, movie.0, movie.1, b"udta")?
    } else {
        None
    };
    let Some(user_data) = user_data else {
        return Ok(Vec::new());
    };
    let Some(meta) = find_box(bytes, user_data.0, user_data.1, b"meta")? else {
        return Ok(Vec::new());
    };
    let children = meta.0.saturating_add(4);
    let Some(keys_range) = find_box(bytes, children, meta.1, b"keys")? else {
        return Ok(Vec::new());
    };
    let keys = parse_isobmff_keys(bytes, keys_range, limits)?;
    let Some(ilst) = find_box(bytes, children, meta.1, b"ilst")? else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    let mut position = ilst.0;
    while position.saturating_add(8) <= ilst.1 {
        let size = usize::try_from(
            read_be_u32(bytes, position).ok_or_else(|| "truncated ilst item".to_owned())?,
        )
        .map_err(|_| "ilst item size does not fit this platform".to_owned())?;
        if size <= 8 || position.saturating_add(size) > ilst.1 {
            return Err("invalid ilst item range".to_owned());
        }
        let index = read_be_u32(bytes, position + 4)
            .ok_or_else(|| "truncated ilst item index".to_owned())?;
        if let Some(key) = keys.get(&index) {
            if let Some(data_range) = find_box(bytes, position + 8, position + size, b"data")? {
                let value_start = data_range.0.saturating_add(8);
                if let Some(value) = bytes.get(value_start..data_range.1) {
                    push_entry(&mut entries, key.clone(), decode_text(value), limits)?;
                }
            }
        }
        position += size;
    }
    Ok(entries)
}

fn parse_isobmff_keys(
    bytes: &[u8],
    range: (usize, usize),
    limits: &MetadataLimits,
) -> Result<BTreeMap<u32, String>, String> {
    let mut position = range.0.saturating_add(4);
    let count = read_be_u32(bytes, position).ok_or_else(|| "truncated keys count".to_owned())?;
    position += 4;
    if usize::try_from(count).unwrap_or(usize::MAX) > limits.max_entries {
        return Err("ISOBMFF key count exceeds the configured limit".to_owned());
    }
    let mut keys = BTreeMap::new();
    for index in 1..=count {
        let size = usize::try_from(
            read_be_u32(bytes, position).ok_or_else(|| "truncated key size".to_owned())?,
        )
        .map_err(|_| "key size does not fit this platform".to_owned())?;
        if size < 8 || position.saturating_add(size) > range.1 {
            return Err("invalid key range".to_owned());
        }
        let name = decode_text(
            bytes
                .get(position + 8..position + size)
                .ok_or_else(|| "truncated key name".to_owned())?,
        );
        keys.insert(index, name);
        position += size;
    }
    Ok(keys)
}

fn parse_glb(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    if !bytes.starts_with(b"glTF") || bytes.len() < 20 {
        return Err("invalid GLB header".to_owned());
    }
    let total =
        usize::try_from(read_le_u32(bytes, 8).ok_or_else(|| "truncated GLB length".to_owned())?)
            .map_err(|_| "GLB length does not fit this platform".to_owned())?;
    if total > bytes.len() {
        return Err("GLB total length exceeds the available bytes".to_owned());
    }
    let json_length = usize::try_from(
        read_le_u32(bytes, 12).ok_or_else(|| "truncated GLB JSON length".to_owned())?,
    )
    .map_err(|_| "GLB JSON length does not fit this platform".to_owned())?;
    if bytes.get(16..20) != Some(b"JSON") {
        return Err("GLB first chunk is not JSON".to_owned());
    }
    let json = bytes
        .get(20..20usize.saturating_add(json_length))
        .ok_or_else(|| "truncated GLB JSON chunk".to_owned())?;
    let (normalized, _) = normalize_json_non_finite(json);
    let value: Value = serde_json::from_slice(&normalized)
        .map_err(|error| format!("invalid GLB JSON chunk: {error}"))?;
    let Some(extras) = value
        .get("asset")
        .and_then(|asset| asset.get("extras"))
        .and_then(Value::as_object)
    else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for key in ["workflow", "prompt"] {
        if let Some(value) = extras.get(key) {
            let value = value
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| value.to_string());
            push_entry(&mut entries, key.to_owned(), value, limits)?;
        }
    }
    Ok(entries)
}

fn parse_safetensors(bytes: &[u8], limits: &MetadataLimits) -> Result<Vec<MetadataEntry>, String> {
    let header_length = usize::try_from(
        read_le_u64(bytes, 0).ok_or_else(|| "truncated safetensors header length".to_owned())?,
    )
    .map_err(|_| "safetensors header length does not fit this platform".to_owned())?;
    if header_length > 4 * 1024 * 1024 || header_length > limits.max_value_bytes {
        return Err("safetensors header exceeds the metadata scan limit".to_owned());
    }
    let header = bytes
        .get(8..8usize.saturating_add(header_length))
        .ok_or_else(|| "truncated safetensors header".to_owned())?;
    let value: Value = serde_json::from_slice(header)
        .map_err(|error| format!("invalid safetensors header JSON: {error}"))?;
    let Some(metadata) = value.get("__metadata__").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut entries = Vec::new();
    for (key, value) in metadata {
        let value = value
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| value.to_string());
        push_entry(&mut entries, key.clone(), value, limits)?;
    }
    Ok(entries)
}

fn write_png(bytes: &[u8], fields: &BTreeMap<String, String>) -> Result<Vec<u8>, MetadataError> {
    if !bytes.starts_with(PNG_SIGNATURE) {
        return Err(MetadataError::InvalidWrite(
            "invalid PNG signature".to_owned(),
        ));
    }
    let mut position = PNG_SIGNATURE.len();
    let mut iend = None;
    while position < bytes.len() {
        let length =
            usize::try_from(read_be_u32(bytes, position).ok_or_else(|| {
                MetadataError::InvalidWrite("truncated PNG chunk length".to_owned())
            })?)
            .map_err(|_| MetadataError::InvalidWrite("invalid PNG chunk length".to_owned()))?;
        let end = position
            .checked_add(12)
            .and_then(|value| value.checked_add(length))
            .ok_or_else(|| MetadataError::InvalidWrite("PNG chunk overflow".to_owned()))?;
        if end > bytes.len() {
            return Err(MetadataError::InvalidWrite(
                "truncated PNG chunk".to_owned(),
            ));
        }
        if bytes.get(position + 4..position + 8) == Some(b"IEND") {
            iend = Some(position);
            break;
        }
        position = end;
    }
    let iend = iend.ok_or_else(|| MetadataError::InvalidWrite("PNG has no IEND".to_owned()))?;
    let mut output = Vec::with_capacity(
        bytes.len().saturating_add(
            fields
                .iter()
                .map(|(key, value)| key.len() + value.len() + 13)
                .sum::<usize>(),
        ),
    );
    output.extend_from_slice(&bytes[..iend]);
    for (key, value) in fields {
        let mut payload = Vec::with_capacity(key.len() + value.len() + 1);
        payload.extend_from_slice(key.as_bytes());
        payload.push(0);
        payload.extend_from_slice(value.as_bytes());
        let length = u32::try_from(payload.len()).map_err(|_| MetadataError::OutputTooLarge)?;
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(b"tEXt");
        output.extend_from_slice(&payload);
        let mut checksum_input = Vec::with_capacity(payload.len() + 4);
        checksum_input.extend_from_slice(b"tEXt");
        checksum_input.extend_from_slice(&payload);
        output.extend_from_slice(&crc32(&checksum_input).to_be_bytes());
    }
    output.extend_from_slice(&bytes[iend..]);
    Ok(output)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn write_svg(bytes: &[u8], fields: &BTreeMap<String, String>) -> Result<Vec<u8>, MetadataError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| MetadataError::InvalidWrite(format!("SVG is not UTF-8: {error}")))?;
    let lower = text.to_ascii_lowercase();
    let svg_start = lower
        .find("<svg")
        .ok_or_else(|| MetadataError::InvalidWrite("SVG opening tag is missing".to_owned()))?;
    let tag_end = text[svg_start..]
        .find('>')
        .map(|offset| svg_start + offset + 1)
        .ok_or_else(|| MetadataError::InvalidWrite("SVG opening tag is incomplete".to_owned()))?;
    let mut object = Map::new();
    for (key, value) in fields {
        object.insert(key.clone(), Value::String(value.clone()));
    }
    let payload = serde_json::to_string(&Value::Object(object))
        .map_err(|error| MetadataError::InvalidWrite(error.to_string()))?;
    if payload.contains("]]>") {
        return Err(MetadataError::InvalidWrite(
            "SVG metadata cannot contain a CDATA terminator".to_owned(),
        ));
    }
    let mut output = String::with_capacity(text.len() + payload.len() + 32);
    output.push_str(&text[..tag_end]);
    output.push_str("<metadata><![CDATA[");
    output.push_str(&payload);
    output.push_str("]]></metadata>");
    output.push_str(&text[tag_end..]);
    Ok(output.into_bytes())
}

fn write_safetensors(
    bytes: &[u8],
    fields: &BTreeMap<String, String>,
    limits: &MetadataLimits,
) -> Result<Vec<u8>, MetadataError> {
    let old_length =
        usize::try_from(read_le_u64(bytes, 0).ok_or_else(|| {
            MetadataError::InvalidWrite("truncated safetensors header".to_owned())
        })?)
        .map_err(|_| MetadataError::InvalidWrite("invalid safetensors header length".to_owned()))?;
    let header = bytes
        .get(8..8usize.saturating_add(old_length))
        .ok_or_else(|| MetadataError::InvalidWrite("truncated safetensors header".to_owned()))?;
    let mut value: Value = serde_json::from_slice(header)
        .map_err(|error| MetadataError::InvalidWrite(format!("invalid header JSON: {error}")))?;
    let object = value.as_object_mut().ok_or_else(|| {
        MetadataError::InvalidWrite("safetensors header root is not an object".to_owned())
    })?;
    let metadata = object
        .entry("__metadata__")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| {
            MetadataError::InvalidWrite("safetensors __metadata__ is not an object".to_owned())
        })?;
    for (key, value) in fields {
        metadata.insert(key.clone(), Value::String(value.clone()));
    }
    let mut new_header = serde_json::to_vec(&value)
        .map_err(|error| MetadataError::InvalidWrite(error.to_string()))?;
    while !new_header.len().is_multiple_of(8) {
        new_header.push(b' ');
    }
    if new_header.len() > limits.max_value_bytes {
        return Err(MetadataError::OutputTooLarge);
    }
    let mut output = Vec::with_capacity(
        bytes
            .len()
            .saturating_sub(old_length)
            .saturating_add(new_header.len()),
    );
    output.extend_from_slice(
        &u64::try_from(new_header.len())
            .map_err(|_| MetadataError::OutputTooLarge)?
            .to_le_bytes(),
    );
    output.extend_from_slice(&new_header);
    output.extend_from_slice(bytes.get(8 + old_length..).unwrap_or_default());
    Ok(output)
}

fn write_glb(bytes: &[u8], fields: &BTreeMap<String, String>) -> Result<Vec<u8>, MetadataError> {
    if !bytes.starts_with(b"glTF") || bytes.get(16..20) != Some(b"JSON") {
        return Err(MetadataError::InvalidWrite("invalid GLB".to_owned()));
    }
    let old_json_length = usize::try_from(
        read_le_u32(bytes, 12)
            .ok_or_else(|| MetadataError::InvalidWrite("truncated GLB JSON length".to_owned()))?,
    )
    .map_err(|_| MetadataError::InvalidWrite("invalid GLB JSON length".to_owned()))?;
    let old_json_end = 20usize
        .checked_add(old_json_length)
        .ok_or(MetadataError::OutputTooLarge)?;
    let old_json = bytes
        .get(20..old_json_end)
        .ok_or_else(|| MetadataError::InvalidWrite("truncated GLB JSON".to_owned()))?;
    let mut value: Value = serde_json::from_slice(old_json)
        .map_err(|error| MetadataError::InvalidWrite(format!("invalid GLB JSON: {error}")))?;
    let root = value
        .as_object_mut()
        .ok_or_else(|| MetadataError::InvalidWrite("GLB JSON root is not an object".to_owned()))?;
    let asset = root
        .entry("asset")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| MetadataError::InvalidWrite("GLB asset is not an object".to_owned()))?;
    let extras = asset
        .entry("extras")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| MetadataError::InvalidWrite("GLB extras is not an object".to_owned()))?;
    for (key, value) in fields {
        extras.insert(key.clone(), Value::String(value.clone()));
    }
    let mut json = serde_json::to_vec(&value)
        .map_err(|error| MetadataError::InvalidWrite(error.to_string()))?;
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 20usize
        .checked_add(json.len())
        .and_then(|length| length.checked_add(bytes.len().saturating_sub(old_json_end)))
        .ok_or(MetadataError::OutputTooLarge)?;
    let mut output = Vec::with_capacity(total);
    output.extend_from_slice(&bytes[..8]);
    output.extend_from_slice(
        &u32::try_from(total)
            .map_err(|_| MetadataError::OutputTooLarge)?
            .to_le_bytes(),
    );
    output.extend_from_slice(
        &u32::try_from(json.len())
            .map_err(|_| MetadataError::OutputTooLarge)?
            .to_le_bytes(),
    );
    output.extend_from_slice(b"JSON");
    output.extend_from_slice(&json);
    output.extend_from_slice(bytes.get(old_json_end..).unwrap_or_default());
    Ok(output)
}

fn write_open_exr(
    bytes: &[u8],
    fields: &BTreeMap<String, String>,
) -> Result<Vec<u8>, MetadataError> {
    if !bytes.starts_with(EXR_SIGNATURE) || bytes.len() < 8 {
        return Err(MetadataError::InvalidWrite("invalid OpenEXR".to_owned()));
    }
    let mut position = 8usize;
    let mut data_window = None;
    let mut compression = 0u8;
    while position < bytes.len() && bytes[position] != 0 {
        let name_end = bytes[position..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|offset| position + offset)
            .ok_or_else(|| {
                MetadataError::InvalidWrite("unterminated EXR attribute name".to_owned())
            })?;
        let type_start = name_end + 1;
        let type_end = bytes[type_start..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|offset| type_start + offset)
            .ok_or_else(|| {
                MetadataError::InvalidWrite("unterminated EXR attribute type".to_owned())
            })?;
        let size = usize::try_from(read_le_u32(bytes, type_end + 1).ok_or_else(|| {
            MetadataError::InvalidWrite("truncated EXR attribute size".to_owned())
        })?)
        .map_err(|_| MetadataError::InvalidWrite("invalid EXR attribute size".to_owned()))?;
        let value_start = type_end + 5;
        let value_end = value_start
            .checked_add(size)
            .ok_or(MetadataError::OutputTooLarge)?;
        let value = bytes.get(value_start..value_end).ok_or_else(|| {
            MetadataError::InvalidWrite("truncated EXR attribute value".to_owned())
        })?;
        let name = decode_text(&bytes[position..name_end]);
        let attribute_type = decode_text(&bytes[type_start..type_end]);
        if name == "dataWindow" && attribute_type == "box2i" && value.len() == 16 {
            let y_min =
                i32::from_le_bytes(value[4..8].try_into().map_err(|_| {
                    MetadataError::InvalidWrite("invalid EXR dataWindow".to_owned())
                })?);
            let y_max =
                i32::from_le_bytes(value[12..16].try_into().map_err(|_| {
                    MetadataError::InvalidWrite("invalid EXR dataWindow".to_owned())
                })?);
            data_window = Some((y_min, y_max));
        } else if name == "compression" && attribute_type == "compression" {
            compression = value.first().copied().unwrap_or(0);
        }
        position = value_end;
    }
    let (y_min, y_max) = data_window.ok_or_else(|| {
        MetadataError::InvalidWrite("OpenEXR dataWindow attribute is missing".to_owned())
    })?;
    if y_max < y_min {
        return Err(MetadataError::InvalidWrite(
            "OpenEXR dataWindow is inverted".to_owned(),
        ));
    }
    let scanlines_per_chunk = match compression {
        3 | 5 => 16usize,
        4 | 6 | 7 => 32,
        8 | 9 => 256,
        _ => 1,
    };
    let height = usize::try_from(i64::from(y_max) - i64::from(y_min) + 1)
        .map_err(|_| MetadataError::InvalidWrite("OpenEXR height is invalid".to_owned()))?;
    let chunk_count = height.div_ceil(scanlines_per_chunk);
    let table_start = position.saturating_add(1);
    let table_end = table_start
        .checked_add(
            chunk_count
                .checked_mul(8)
                .ok_or(MetadataError::OutputTooLarge)?,
        )
        .ok_or(MetadataError::OutputTooLarge)?;
    let old_table = bytes
        .get(table_start..table_end)
        .ok_or_else(|| MetadataError::InvalidWrite("truncated OpenEXR chunk table".to_owned()))?;
    let mut attributes = Vec::new();
    for (key, value) in fields {
        attributes.extend_from_slice(key.as_bytes());
        attributes.push(0);
        attributes.extend_from_slice(b"string\0");
        attributes.extend_from_slice(
            &u32::try_from(value.len())
                .map_err(|_| MetadataError::OutputTooLarge)?
                .to_le_bytes(),
        );
        attributes.extend_from_slice(value.as_bytes());
    }
    let delta = u64::try_from(attributes.len()).map_err(|_| MetadataError::OutputTooLarge)?;
    let mut output = Vec::with_capacity(bytes.len().saturating_add(attributes.len()));
    output.extend_from_slice(&bytes[..position]);
    output.extend_from_slice(&attributes);
    output.extend_from_slice(&bytes[position..table_start]);
    for chunk in old_table.chunks_exact(8) {
        let offset =
            u64::from_le_bytes(chunk.try_into().map_err(|_| {
                MetadataError::InvalidWrite("invalid OpenEXR chunk offset".to_owned())
            })?);
        output.extend_from_slice(
            &offset
                .checked_add(delta)
                .ok_or(MetadataError::OutputTooLarge)?
                .to_le_bytes(),
        );
    }
    output.extend_from_slice(bytes.get(table_end..).unwrap_or_default());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::ZlibEncoder};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{fs, io::Write, path::PathBuf};

    fn minimal_png() -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(b"IEND");
        bytes.extend_from_slice(&crc32(b"IEND").to_be_bytes());
        bytes
    }

    fn fields() -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "prompt".to_owned(),
                r#"{"1":{"class_type":"A","inputs":{}}}"#.to_owned(),
            ),
            (
                "workflow".to_owned(),
                r#"{"version":0.4,"future":true}"#.to_owned(),
            ),
        ])
    }

    fn png_chunk(chunk_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        chunk.extend_from_slice(chunk_type);
        chunk.extend_from_slice(payload);
        let mut checksum = chunk_type.to_vec();
        checksum.extend_from_slice(payload);
        chunk.extend_from_slice(&crc32(&checksum).to_be_bytes());
        chunk
    }

    #[test]
    fn png_source_pixel_facts_are_checked_cancelled_and_stop_at_image_data() {
        let header = [0, 0, 0, 5, 0, 0, 0, 1, 16, 0, 0, 0, 0];
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend(png_chunk(b"IHDR", &header));
        let header_end = bytes.len();
        bytes.extend(png_chunk(b"tRNS", &256_u16.to_be_bytes()));
        bytes.extend(png_chunk(b"IDAT", &[]));
        bytes.extend(b"deliberately unparsed image tail");
        let cancellation = sim_tensor::CancellationToken::default();
        let format = png_source_pixel_format(&bytes, &cancellation);
        assert!(matches!(
            format,
            Ok(PngSourcePixelFormat {
                bit_depth: 16,
                color_type: 0,
                transparency: Some(PngSourceTransparency::Grayscale(256)),
            }),
        ));
        for length in [0, 7, 12, header_end - 1, header_end + 9] {
            assert!(matches!(
                png_source_pixel_format(&bytes[..length], &cancellation),
                Err(PngSourcePixelError::Malformed(_)),
            ));
        }
        let mut malformed = bytes[..header_end].to_vec();
        malformed.extend(png_chunk(b"tRNS", &[1]));
        assert!(matches!(
            png_source_pixel_format(&malformed, &cancellation),
            Err(PngSourcePixelError::Malformed(_)),
        ));
        cancellation.cancel();
        assert!(matches!(
            png_source_pixel_format(&bytes, &cancellation),
            Err(PngSourcePixelError::Cancelled),
        ));
    }

    fn compressed_itxt_png() -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(br#"{"version":0.4,"nodes":[],"links":[]}"#)
            .expect("zlib payload");
        let compressed = encoder.finish().expect("zlib completion");
        let mut payload = b"workflow\0\x01\x00\0\0".to_vec();
        payload.extend_from_slice(&compressed);
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&png_chunk(b"iTXt", &payload));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        png
    }

    fn png_with_bad_itxt_then_prompt() -> Vec<u8> {
        let mut bad_itxt = b"workflow\0\x01\x00\0\0".to_vec();
        bad_itxt.extend_from_slice(b"not-a-zlib-stream");
        let prompt = b"prompt\0{\"1\":{\"class_type\":\"KSampler\",\"inputs\":{}}}";
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&png_chunk(b"iTXt", &bad_itxt));
        png.extend_from_slice(&png_chunk(b"tEXt", prompt));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        png
    }

    fn animated_comf_png() -> Vec<u8> {
        let mut payload = b"workflow\0".to_vec();
        payload.extend_from_slice(br#"{"version":0.4,"nodes":[],"links":[]}"#);
        let mut png = PNG_SIGNATURE.to_vec();
        png.extend_from_slice(&png_chunk(b"comf", &payload));
        png.extend_from_slice(&png_chunk(b"IEND", &[]));
        png
    }

    fn tiff_with_workflow() -> Vec<u8> {
        let text = b"workflow:{\"version\":0.4,\"nodes\":[],\"links\":[]}\0";
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&270u16.to_le_bytes());
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&(text.len() as u32).to_le_bytes());
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(text);
        tiff
    }

    fn tiff_with_usercomment() -> Vec<u8> {
        let text = b"ASCII\0\0\0{\"prompt\":{\"1\":{\"class_type\":\"KSampler\",\"inputs\":{\"cfg\":NaN}}},\"workflow\":{\"version\":0.4,\"nodes\":[],\"links\":[]}}\0";
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II");
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x9286u16.to_le_bytes());
        tiff.extend_from_slice(&7u16.to_le_bytes());
        tiff.extend_from_slice(&(text.len() as u32).to_le_bytes());
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(text);
        tiff
    }

    fn webp_fixture() -> Vec<u8> {
        let mut payload = b"Exif\0\0".to_vec();
        payload.extend_from_slice(&tiff_with_workflow());
        let mut webp = b"RIFF".to_vec();
        let riff_size = 4usize + 8 + payload.len() + payload.len() % 2;
        webp.extend_from_slice(&(riff_size as u32).to_le_bytes());
        webp.extend_from_slice(b"WEBP");
        webp.extend_from_slice(b"EXIF");
        webp.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        webp.extend_from_slice(&payload);
        if !payload.len().is_multiple_of(2) {
            webp.push(0);
        }
        webp
    }

    fn vorbis_comments() -> Vec<u8> {
        let vendor = b"zed";
        let comment = b"workflow={\"version\":0.4,\"nodes\":[],\"links\":[]}";
        let mut payload = Vec::new();
        payload.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        payload.extend_from_slice(vendor);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        payload.extend_from_slice(comment);
        payload
    }

    fn flac_fixture() -> Vec<u8> {
        let comments = vorbis_comments();
        let mut bytes = b"fLaC".to_vec();
        bytes.push(0x84);
        let length = comments.len() as u32;
        bytes.extend_from_slice(&length.to_be_bytes()[1..]);
        bytes.extend_from_slice(&comments);
        bytes
    }

    fn ogg_fixture() -> Vec<u8> {
        let mut packet = b"OpusTags".to_vec();
        packet.extend_from_slice(&vorbis_comments());
        let mut page = vec![0u8; 27];
        page[..4].copy_from_slice(b"OggS");
        page[26] = 1;
        page.push(packet.len() as u8);
        page.extend_from_slice(&packet);
        page
    }

    fn webm_fixture() -> Vec<u8> {
        let name = b"workflow";
        let value = br#"{"version":0.4,"nodes":[],"links":[]}"#;
        let mut payload = vec![0x45, 0xa3, 0x80 | name.len() as u8];
        payload.extend_from_slice(name);
        payload.extend_from_slice(&[0x44, 0x87, 0x80 | value.len() as u8]);
        payload.extend_from_slice(value);
        let mut bytes = vec![0x1a, 0x45, 0xdf, 0xa3, 0x80, 0x67, 0xc8];
        bytes.push(0x80 | payload.len() as u8);
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn isobmff_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(&((payload.len() + 8) as u32).to_be_bytes());
        output.extend_from_slice(box_type);
        output.extend_from_slice(payload);
        output
    }

    fn isobmff_fixture() -> Vec<u8> {
        let ftyp = isobmff_box(b"ftyp", b"isom\0\0\0\0isom");
        let key = {
            let mut payload = b"mdta".to_vec();
            payload.extend_from_slice(b"workflow");
            let mut entry = ((payload.len() + 4) as u32).to_be_bytes().to_vec();
            entry.extend_from_slice(&payload);
            entry
        };
        let mut keys_payload = vec![0, 0, 0, 0];
        keys_payload.extend_from_slice(&1u32.to_be_bytes());
        keys_payload.extend_from_slice(&key);
        let keys = isobmff_box(b"keys", &keys_payload);
        let mut data_payload = vec![0; 8];
        data_payload.extend_from_slice(br#"{"version":0.4,"nodes":[],"links":[]}"#);
        let data = isobmff_box(b"data", &data_payload);
        let item = isobmff_box(&1u32.to_be_bytes(), &data);
        let ilst = isobmff_box(b"ilst", &item);
        let mut meta_payload = vec![0; 4];
        meta_payload.extend_from_slice(&keys);
        meta_payload.extend_from_slice(&ilst);
        let meta = isobmff_box(b"meta", &meta_payload);
        let udta = isobmff_box(b"udta", &meta);
        let moov = isobmff_box(b"moov", &udta);
        [ftyp, moov].concat()
    }

    fn glb_fixture() -> Vec<u8> {
        let mut json = br#"{"asset":{"version":"2.0","extras":{"workflow":{"version":0.4,"nodes":[],"links":[]}}}}"#.to_vec();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 20 + json.len();
        let mut glb = b"glTF".to_vec();
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        glb
    }

    fn glb_nonfinite_fixture() -> Vec<u8> {
        let mut json = br#"{"asset":{"version":"2.0","extras":{"prompt":{"1":{"class_type":"KSampler","inputs":{"cfg":NaN,"denoise":Infinity}}}}}}"#.to_vec();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 20 + json.len();
        let mut glb = b"glTF".to_vec();
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        glb
    }

    fn avif_with_tiff(tiff: &[u8]) -> Vec<u8> {
        let ftyp = isobmff_box(b"ftyp", b"avif\0\0\0\0avif");
        let mut infe_payload = vec![2, 0, 0, 0];
        infe_payload.extend_from_slice(&1u16.to_be_bytes());
        infe_payload.extend_from_slice(&0u16.to_be_bytes());
        infe_payload.extend_from_slice(b"Exif");
        infe_payload.extend_from_slice(b"metadata\0");
        let infe = isobmff_box(b"infe", &infe_payload);
        let mut iinf_payload = vec![0, 0, 0, 0];
        iinf_payload.extend_from_slice(&1u16.to_be_bytes());
        iinf_payload.extend_from_slice(&infe);
        let iinf = isobmff_box(b"iinf", &iinf_payload);
        let mut exif = 0u32.to_be_bytes().to_vec();
        exif.extend_from_slice(tiff);
        let placeholder_iloc = {
            let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&0u16.to_be_bytes());
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&0u32.to_be_bytes());
            payload.extend_from_slice(&(exif.len() as u32).to_be_bytes());
            isobmff_box(b"iloc", &payload)
        };
        let mut meta_payload = vec![0; 4];
        meta_payload.extend_from_slice(&iinf);
        meta_payload.extend_from_slice(&placeholder_iloc);
        let placeholder_meta = isobmff_box(b"meta", &meta_payload);
        let extent_offset = ftyp.len() + placeholder_meta.len() + 8;
        let iloc = {
            let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&0u16.to_be_bytes());
            payload.extend_from_slice(&1u16.to_be_bytes());
            payload.extend_from_slice(&(extent_offset as u32).to_be_bytes());
            payload.extend_from_slice(&(exif.len() as u32).to_be_bytes());
            isobmff_box(b"iloc", &payload)
        };
        let mut meta_payload = vec![0; 4];
        meta_payload.extend_from_slice(&iinf);
        meta_payload.extend_from_slice(&iloc);
        let meta = isobmff_box(b"meta", &meta_payload);
        let mdat = isobmff_box(b"mdat", &exif);
        [ftyp, meta, mdat].concat()
    }

    fn avif_fixture() -> Vec<u8> {
        avif_with_tiff(&tiff_with_workflow())
    }

    fn carrier_fixtures() -> Vec<(&'static str, Vec<u8>, &'static str)> {
        vec![
            ("png-itxt", compressed_itxt_png(), "fixture.png"),
            ("apng-comf", animated_comf_png(), "fixture.png"),
            ("webp", webp_fixture(), "fixture.webp"),
            ("avif", avif_fixture(), "fixture.avif"),
            ("svg", br#"<svg><metadata><![CDATA[{"workflow":{"version":0.4,"nodes":[],"links":[]}}]]></metadata></svg>"#.to_vec(), "fixture.svg"),
            ("flac", flac_fixture(), "fixture.flac"),
            ("mp3", b"ID3prompt\0{\"1\":{\"class_type\":\"A\",\"inputs\":{}}}\0\xff\xfb".to_vec(), "fixture.mp3"),
            ("ogg-opus", ogg_fixture(), "fixture.opus"),
            ("webm", webm_fixture(), "fixture.webm"),
            ("isobmff", isobmff_fixture(), "fixture.mp4"),
            ("glb", glb_fixture(), "fixture.glb"),
        ]
    }

    fn carrier_results() -> BTreeMap<&'static str, bool> {
        let limits = MetadataLimits::default();
        carrier_fixtures()
            .into_iter()
            .map(|(name, bytes, file_name)| {
                let document =
                    MetadataDocument::parse(&bytes, Some(file_name), None, limits.clone())
                        .expect("bounded carrier fixture");
                let found = document.get_case_insensitive("workflow").is_some()
                    || document.get_case_insensitive("prompt").is_some();
                (name, found && document.original_bytes() == bytes)
            })
            .collect()
    }

    fn carrier_digests() -> BTreeMap<&'static str, String> {
        carrier_fixtures()
            .into_iter()
            .map(|(name, bytes, _)| (name, format!("{:x}", Sha256::digest(bytes))))
            .collect()
    }

    struct FrontendFixture {
        name: &'static str,
        file_name: &'static str,
        bytes: Vec<u8>,
        carrier: MetadataCarrier,
        has_workflow: bool,
        has_prompt: bool,
        has_nonfinite_prompt: bool,
    }

    fn frontend_fixture_bytes(file_name: &str) -> Vec<u8> {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("projects")
            .join("sim")
            .join("ComfyUI-Frontend/src/scripts/metadata/__fixtures__")
            .join(file_name);
        fs::read(&fixture_path).unwrap_or_else(|error| {
            panic!(
                "failed to read frontend metadata fixture {}: {error}",
                fixture_path.display()
            )
        })
    }

    fn frontend_fixture(
        name: &'static str,
        file_name: &'static str,
        carrier: MetadataCarrier,
        has_workflow: bool,
        has_prompt: bool,
        has_nonfinite_prompt: bool,
    ) -> FrontendFixture {
        FrontendFixture {
            name,
            file_name,
            bytes: frontend_fixture_bytes(file_name),
            carrier,
            has_workflow,
            has_prompt,
            has_nonfinite_prompt,
        }
    }

    fn frontend_fixtures() -> [FrontendFixture; 17] {
        [
            frontend_fixture(
                "frontend-png",
                "with_metadata.png",
                MetadataCarrier::Png,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-png-nonfinite",
                "with_nan_metadata.png",
                MetadataCarrier::Png,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-webp",
                "with_metadata.webp",
                MetadataCarrier::WebP,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-webp-exif-prefix",
                "with_metadata_exif_prefix.webp",
                MetadataCarrier::WebP,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-webp-nonfinite",
                "with_nan_metadata.webp",
                MetadataCarrier::WebP,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-avif",
                "with_metadata.avif",
                MetadataCarrier::Avif,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-avif-nonfinite",
                "with_nan_metadata.avif",
                MetadataCarrier::Avif,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-flac",
                "with_metadata.flac",
                MetadataCarrier::Flac,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-flac-nonfinite",
                "with_nan_metadata.flac",
                MetadataCarrier::Flac,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-mp3",
                "with_metadata.mp3",
                MetadataCarrier::Mp3,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-mp3-nonfinite",
                "with_nan_metadata.mp3",
                MetadataCarrier::Mp3,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-opus",
                "with_metadata.opus",
                MetadataCarrier::OggOpus,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-opus-nonfinite",
                "with_nan_metadata.opus",
                MetadataCarrier::OggOpus,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-webm",
                "with_metadata.webm",
                MetadataCarrier::WebM,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-webm-nonfinite",
                "with_nan_metadata.webm",
                MetadataCarrier::WebM,
                false,
                true,
                true,
            ),
            frontend_fixture(
                "frontend-mp4",
                "with_metadata.mp4",
                MetadataCarrier::IsobmffVideo,
                true,
                true,
                false,
            ),
            frontend_fixture(
                "frontend-mp4-nonfinite",
                "with_nan_metadata.mp4",
                MetadataCarrier::IsobmffVideo,
                false,
                true,
                true,
            ),
        ]
    }

    fn open_exr_fixture() -> (Vec<u8>, u64) {
        fn attribute(name: &str, attribute_type: &str, value: &[u8]) -> Vec<u8> {
            let mut bytes = name.as_bytes().to_vec();
            bytes.push(0);
            bytes.extend_from_slice(attribute_type.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value);
            bytes
        }

        let mut exr = EXR_SIGNATURE.to_vec();
        exr.extend_from_slice(&2u32.to_le_bytes());
        let mut window = Vec::new();
        for coordinate in [0i32, 0, 0, 0] {
            window.extend_from_slice(&coordinate.to_le_bytes());
        }
        exr.extend_from_slice(&attribute("dataWindow", "box2i", &window));
        exr.extend_from_slice(&attribute("compression", "compression", &[0]));
        exr.push(0);
        let old_offset = exr.len() as u64 + 8;
        exr.extend_from_slice(&old_offset.to_le_bytes());
        exr.extend_from_slice(b"pixels");
        (exr, old_offset)
    }

    #[test]
    fn png_metadata_round_trips_and_preserves_original() {
        let original = minimal_png();
        let document = MetadataDocument::parse(
            &original,
            Some("fixture.png"),
            None,
            MetadataLimits::default(),
        )
        .expect("bounded document");
        assert_eq!(document.original_bytes(), original);
        let output = document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("metadata write");
        let reopened = MetadataDocument::parse(
            &output,
            Some("fixture.png"),
            None,
            MetadataLimits::default(),
        )
        .expect("reopened document");
        assert_eq!(
            reopened.get_case_insensitive("Workflow"),
            fields().get("workflow").map(String::as_str)
        );
        assert!(reopened.diagnostics().is_empty());
    }

    #[test]
    fn malformed_and_oversized_inputs_are_visible_and_non_panicking() {
        let malformed = MetadataDocument::parse(
            b"\x89PNG\r\n\x1a\n\0",
            None,
            Some("image/png"),
            MetadataLimits::default(),
        )
        .expect("malformed content remains inspectable");
        assert_eq!(malformed.entries(), &[]);
        assert_eq!(malformed.diagnostics().len(), 1);

        let limits = MetadataLimits {
            max_input_bytes: 4,
            ..MetadataLimits::default()
        };
        assert!(matches!(
            MetadataDocument::parse(b"12345", None, None, limits),
            Err(MetadataError::InputTooLarge { .. })
        ));
    }

    #[test]
    fn safetensors_preserves_unknown_header_and_tensor_bytes() {
        let header = br#"{"__metadata__":{"future":"keep"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let document = MetadataDocument::parse(
            &bytes,
            Some("fixture.safetensors"),
            None,
            MetadataLimits::default(),
        )
        .expect("safetensors document");
        let output = document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("safetensors metadata write");
        assert!(output.ends_with(&[1, 2, 3, 4]));
        let reopened = MetadataDocument::parse(
            &output,
            Some("fixture.safetensors"),
            None,
            MetadataLimits::default(),
        )
        .expect("reopened safetensors");
        assert_eq!(reopened.get_case_insensitive("future"), Some("keep"));
        assert_eq!(
            reopened.get_case_insensitive("workflow"),
            fields().get("workflow").map(String::as_str)
        );
    }

    #[test]
    fn glb_metadata_round_trip_preserves_asset_and_binary_chunks() {
        let mut original = glb_fixture();
        original.extend_from_slice(&4u32.to_le_bytes());
        original.extend_from_slice(b"BIN\0");
        original.extend_from_slice(&[1, 2, 3, 4]);
        let total = u32::try_from(original.len()).expect("fixture length");
        original[8..12].copy_from_slice(&total.to_le_bytes());
        let original_json_length = usize::try_from(u32::from_le_bytes(
            original[12..16].try_into().expect("JSON length"),
        ))
        .expect("JSON length fits");
        let original_binary = original[20 + original_json_length..].to_vec();

        let document = MetadataDocument::parse(
            &original,
            Some("fixture.glb"),
            None,
            MetadataLimits::default(),
        )
        .expect("GLB document");
        let output = document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("GLB metadata write");
        assert!(output.ends_with(&original_binary));
        assert_eq!(
            usize::try_from(u32::from_le_bytes(
                output[8..12].try_into().expect("total length")
            ))
            .expect("total length fits"),
            output.len()
        );
        let reopened = MetadataDocument::parse(
            &output,
            Some("fixture.glb"),
            None,
            MetadataLimits::default(),
        )
        .expect("reopened GLB");
        assert_eq!(
            reopened.get_case_insensitive("workflow"),
            fields().get("workflow").map(String::as_str)
        );
        let json_length = usize::try_from(u32::from_le_bytes(
            output[12..16].try_into().expect("JSON length"),
        ))
        .expect("JSON length fits");
        let json: Value =
            serde_json::from_slice(&output[20..20 + json_length]).expect("rewritten GLB JSON");
        assert_eq!(json["asset"]["version"], "2.0");
    }

    #[test]
    fn val_metadata_001_svg_and_glb_use_the_canonical_nonfinite_json_compatibility() {
        let svg = br#"<svg><metadata><![CDATA[{"prompt":{"1":{"class_type":"KSampler","inputs":{"cfg":NaN,"denoise":Infinity}}}}]]></metadata></svg>"#;
        let svg_document =
            MetadataDocument::parse(svg, Some("nonfinite.svg"), None, MetadataLimits::default())
                .expect("non-finite SVG metadata");
        let svg_prompt: Value = serde_json::from_str(
            svg_document
                .get_case_insensitive("prompt")
                .expect("SVG prompt"),
        )
        .expect("normalized SVG prompt");
        assert!(svg_prompt["1"]["inputs"]["cfg"].is_null());
        assert!(svg_prompt["1"]["inputs"]["denoise"].is_null());

        let glb = glb_nonfinite_fixture();
        let glb_document =
            MetadataDocument::parse(&glb, Some("nonfinite.glb"), None, MetadataLimits::default())
                .expect("non-finite GLB metadata");
        let glb_prompt: Value = serde_json::from_str(
            glb_document
                .get_case_insensitive("prompt")
                .expect("GLB prompt"),
        )
        .expect("normalized GLB prompt");
        assert_eq!(glb_prompt, svg_prompt);
        assert!(svg_document.diagnostics().is_empty());
        assert!(glb_document.diagnostics().is_empty());
    }

    #[test]
    fn val_metadata_001_avif_usercomment_object_uses_the_shared_json_adapter() {
        let avif = avif_with_tiff(&tiff_with_usercomment());
        let document = MetadataDocument::parse(
            &avif,
            Some("usercomment.avif"),
            None,
            MetadataLimits::default(),
        )
        .expect("AVIF usercomment metadata");
        let prompt: Value = serde_json::from_str(
            document
                .get_case_insensitive("prompt")
                .expect("AVIF usercomment prompt"),
        )
        .expect("normalized AVIF usercomment prompt");
        assert!(prompt["1"]["inputs"]["cfg"].is_null());
        assert!(document.get_case_insensitive("workflow").is_some());
        assert!(document.diagnostics().is_empty());
        assert_eq!(document.original_bytes(), avif);
    }

    #[test]
    fn val_metadata_001_png_bad_itxt_skips_only_that_chunk() {
        let png = png_with_bad_itxt_then_prompt();
        let document =
            MetadataDocument::parse(&png, Some("bad-itxt.png"), None, MetadataLimits::default())
                .expect("malformed iTXt remains inspectable");
        assert!(document.get_case_insensitive("workflow").is_none());
        assert!(document.get_case_insensitive("prompt").is_some());
        assert_eq!(document.diagnostics().len(), 1);
        assert!(
            document.diagnostics()[0]
                .message
                .starts_with("iTXt decompression failed:")
        );
        assert_eq!(document.original_bytes(), png);
    }

    #[test]
    fn carrier_detection_uses_content_before_extension() {
        assert_eq!(
            detect_carrier(&minimal_png(), Some("wrong.mp4"), Some("video/mp4")),
            MetadataCarrier::Png
        );
        assert_eq!(
            detect_carrier(b"ply\nformat ascii 1.0\n", Some("asset.bin"), None),
            MetadataCarrier::Ply
        );
    }

    #[test]
    fn disable_metadata_matrix_keeps_source_svg_exception() {
        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#;
        let document =
            MetadataDocument::parse(svg, Some("fixture.svg"), None, MetadataLimits::default())
                .expect("SVG document");
        let output = document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy {
                    metadata_enabled: false,
                },
                &MetadataLimits::default(),
            )
            .expect("source-compatible SVG metadata write");
        assert_ne!(output, svg);

        let png = minimal_png();
        let document =
            MetadataDocument::parse(&png, Some("fixture.png"), None, MetadataLimits::default())
                .expect("PNG document");
        assert_eq!(
            document
                .embed_sim_metadata(
                    &fields(),
                    MetadataWritePolicy {
                        metadata_enabled: false,
                    },
                    &MetadataLimits::default(),
                )
                .expect("suppressed PNG metadata"),
            png
        );

        let glb = glb_fixture();
        let glb_document =
            MetadataDocument::parse(&glb, Some("fixture.glb"), None, MetadataLimits::default())
                .expect("GLB document");
        assert_eq!(
            glb_document
                .embed_sim_metadata(
                    &fields(),
                    MetadataWritePolicy {
                        metadata_enabled: false,
                    },
                    &MetadataLimits::default(),
                )
                .expect("suppressed GLB metadata"),
            glb
        );

        let safetensors_header = br#"{"__metadata__":{"future":"keep"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(safetensors_header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(safetensors_header);
        safetensors.extend_from_slice(&[1, 2, 3, 4]);
        let safetensors_document = MetadataDocument::parse(
            &safetensors,
            Some("fixture.safetensors"),
            None,
            MetadataLimits::default(),
        )
        .expect("safetensors document");
        assert_eq!(
            safetensors_document
                .embed_sim_metadata(
                    &fields(),
                    MetadataWritePolicy {
                        metadata_enabled: false,
                    },
                    &MetadataLimits::default(),
                )
                .expect("suppressed safetensors metadata"),
            safetensors
        );

        let (exr, _) = open_exr_fixture();
        let exr_document =
            MetadataDocument::parse(&exr, Some("fixture.exr"), None, MetadataLimits::default())
                .expect("EXR document");
        assert_eq!(
            exr_document
                .embed_sim_metadata(
                    &fields(),
                    MetadataWritePolicy {
                        metadata_enabled: false,
                    },
                    &MetadataLimits::default(),
                )
                .expect("suppressed EXR metadata"),
            exr
        );
    }

    #[test]
    fn every_cataloged_readable_carrier_has_a_bounded_fixture() {
        let results = carrier_results();
        assert_eq!(results.len(), 11);
        assert!(results.values().all(|passed| *passed), "{results:?}");
        let ply = MetadataDocument::parse(
            b"ply\nformat ascii 1.0\nend_header\n",
            Some("fixture.ply"),
            None,
            MetadataLimits::default(),
        )
        .expect("PLY boundary");
        assert_eq!(ply.support(), MetadataSupport::ExplicitNonCarrier);
        assert!(ply.entries().is_empty());
    }

    #[test]
    fn val_metadata_001_frontend_fixtures_match_the_native_carrier_adapter() {
        for fixture in frontend_fixtures() {
            let document = MetadataDocument::parse(
                &fixture.bytes,
                Some(fixture.file_name),
                None,
                MetadataLimits::default(),
            )
            .expect("frontend metadata fixture");
            assert_eq!(document.carrier(), fixture.carrier, "{}", fixture.name);
            assert_eq!(
                detect_carrier(
                    &fixture.bytes,
                    Some("misleading.json"),
                    Some("application/json")
                ),
                fixture.carrier,
                "{} content detection",
                fixture.name
            );
            assert_eq!(
                document.get_case_insensitive("workflow").is_some(),
                fixture.has_workflow,
                "{} workflow projection",
                fixture.name
            );
            assert_eq!(
                document.get_case_insensitive("prompt").is_some(),
                fixture.has_prompt,
                "{} prompt projection",
                fixture.name
            );
            assert_eq!(
                document
                    .get_case_insensitive("prompt")
                    .is_some_and(|prompt| prompt.contains("NaN") && prompt.contains("Infinity")),
                fixture.has_nonfinite_prompt,
                "{} raw non-finite adapter boundary",
                fixture.name
            );
            assert_eq!(
                document.original_bytes(),
                fixture.bytes.as_slice(),
                "{}",
                fixture.name
            );
            assert!(document.diagnostics().is_empty(), "{}", fixture.name);
        }
    }

    #[test]
    fn open_exr_writer_shifts_absolute_chunk_offsets() {
        let (exr, old_offset) = open_exr_fixture();
        let document =
            MetadataDocument::parse(&exr, Some("fixture.exr"), None, MetadataLimits::default())
                .expect("EXR document");
        assert_eq!(document.support(), MetadataSupport::WriteOnly);
        let output = document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("EXR metadata write");
        let delta = output.len() - exr.len();
        let table_position = output.len() - b"pixels".len() - 8;
        let shifted = u64::from_le_bytes(
            output[table_position..table_position + 8]
                .try_into()
                .expect("chunk offset"),
        );
        assert_eq!(shifted, old_offset + delta as u64);
        assert!(output.ends_with(b"pixels"));
    }

    #[test]
    fn val_metadata_001() {
        let mut cases = carrier_results()
            .into_iter()
            .map(|(name, passed)| (format!("carrier_{name}"), passed))
            .collect::<BTreeMap<_, _>>();
        let mut source_fixture_digests = BTreeMap::new();
        for fixture in frontend_fixtures() {
            let document = MetadataDocument::parse(
                &fixture.bytes,
                Some(fixture.file_name),
                None,
                MetadataLimits::default(),
            )
            .expect("frontend metadata fixture");
            cases.insert(
                fixture.name.to_owned(),
                document.carrier() == fixture.carrier
                    && detect_carrier(
                        &fixture.bytes,
                        Some("misleading.json"),
                        Some("application/json"),
                    ) == fixture.carrier
                    && document.get_case_insensitive("workflow").is_some() == fixture.has_workflow
                    && document.get_case_insensitive("prompt").is_some() == fixture.has_prompt
                    && document
                        .get_case_insensitive("prompt")
                        .is_some_and(|prompt| {
                            prompt.contains("NaN") && prompt.contains("Infinity")
                        })
                        == fixture.has_nonfinite_prompt
                    && document.original_bytes() == fixture.bytes.as_slice()
                    && document.diagnostics().is_empty(),
            );
            source_fixture_digests.insert(
                fixture.name.to_owned(),
                format!("{:x}", Sha256::digest(&fixture.bytes)),
            );
        }
        let direct_json = frontend_fixture_bytes("with_nan_metadata.json");
        let direct_json_document = MetadataDocument::parse(
            &direct_json,
            Some("with_nan_metadata.json"),
            Some("application/json"),
            MetadataLimits::default(),
        )
        .expect("direct JSON workflow-owner boundary");
        cases.insert(
            "direct_json_is_lossless_workflow_owner_boundary".to_owned(),
            direct_json_document.carrier() == MetadataCarrier::Json
                && direct_json_document.support() == MetadataSupport::Unsupported
                && direct_json_document.original_bytes() == direct_json.as_slice()
                && direct_json_document.entries().is_empty()
                && direct_json_document.diagnostics().is_empty(),
        );
        source_fixture_digests.insert(
            "frontend-direct-json-nonfinite".to_owned(),
            format!("{:x}", Sha256::digest(&direct_json)),
        );
        let nonfinite_svg = br#"<svg><metadata><![CDATA[{"prompt":{"1":{"class_type":"KSampler","inputs":{"cfg":NaN,"denoise":Infinity}}}}]]></metadata></svg>"#;
        let nonfinite_svg_document = MetadataDocument::parse(
            nonfinite_svg,
            Some("nonfinite.svg"),
            None,
            MetadataLimits::default(),
        )
        .expect("non-finite SVG metadata");
        let nonfinite_svg_prompt = nonfinite_svg_document
            .get_case_insensitive("prompt")
            .and_then(|prompt| serde_json::from_str::<Value>(prompt).ok());
        cases.insert(
            "svg_nonfinite_uses_canonical_json_compatibility".to_owned(),
            nonfinite_svg_prompt.is_some_and(|prompt| {
                prompt["1"]["inputs"]["cfg"].is_null() && prompt["1"]["inputs"]["denoise"].is_null()
            }) && nonfinite_svg_document.original_bytes() == nonfinite_svg
                && nonfinite_svg_document.diagnostics().is_empty(),
        );
        let nonfinite_glb = glb_nonfinite_fixture();
        let nonfinite_glb_document = MetadataDocument::parse(
            &nonfinite_glb,
            Some("nonfinite.glb"),
            None,
            MetadataLimits::default(),
        )
        .expect("non-finite GLB metadata");
        let nonfinite_glb_prompt = nonfinite_glb_document
            .get_case_insensitive("prompt")
            .and_then(|prompt| serde_json::from_str::<Value>(prompt).ok());
        cases.insert(
            "glb_nonfinite_uses_canonical_json_compatibility".to_owned(),
            nonfinite_glb_prompt.is_some_and(|prompt| {
                prompt["1"]["inputs"]["cfg"].is_null() && prompt["1"]["inputs"]["denoise"].is_null()
            }) && nonfinite_glb_document.original_bytes() == nonfinite_glb
                && nonfinite_glb_document.diagnostics().is_empty(),
        );
        let avif_usercomment = avif_with_tiff(&tiff_with_usercomment());
        let avif_usercomment_document = MetadataDocument::parse(
            &avif_usercomment,
            Some("usercomment.avif"),
            None,
            MetadataLimits::default(),
        )
        .expect("AVIF usercomment metadata");
        let avif_usercomment_prompt = avif_usercomment_document
            .get_case_insensitive("prompt")
            .and_then(|prompt| serde_json::from_str::<Value>(prompt).ok());
        cases.insert(
            "avif_usercomment_object_uses_canonical_json_compatibility".to_owned(),
            avif_usercomment_prompt.is_some_and(|prompt| prompt["1"]["inputs"]["cfg"].is_null())
                && avif_usercomment_document
                    .get_case_insensitive("workflow")
                    .is_some()
                && avif_usercomment_document.original_bytes() == avif_usercomment
                && avif_usercomment_document.diagnostics().is_empty(),
        );
        let bad_itxt_png = png_with_bad_itxt_then_prompt();
        let bad_itxt_document = MetadataDocument::parse(
            &bad_itxt_png,
            Some("bad-itxt.png"),
            None,
            MetadataLimits::default(),
        )
        .expect("malformed iTXt remains inspectable");
        cases.insert(
            "png_bad_itxt_skips_only_bad_chunk".to_owned(),
            bad_itxt_document.get_case_insensitive("workflow").is_none()
                && bad_itxt_document.get_case_insensitive("prompt").is_some()
                && bad_itxt_document.diagnostics().len() == 1
                && bad_itxt_document.diagnostics()[0]
                    .message
                    .starts_with("iTXt decompression failed:")
                && bad_itxt_document.original_bytes() == bad_itxt_png,
        );

        let png = minimal_png();
        let png_document =
            MetadataDocument::parse(&png, Some("fixture.png"), None, MetadataLimits::default())
                .expect("PNG fixture");
        let png_output = png_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("PNG metadata write");
        let png_reopened = MetadataDocument::parse(
            &png_output,
            Some("fixture.png"),
            None,
            MetadataLimits::default(),
        )
        .expect("PNG metadata reopen");
        cases.insert(
            "png_write_round_trip".to_owned(),
            png_reopened.get_case_insensitive("workflow")
                == fields().get("workflow").map(String::as_str),
        );
        let mut png_with_unknown_chunk = PNG_SIGNATURE.to_vec();
        let unknown_chunk = png_chunk(b"vpAg", b"future-payload");
        png_with_unknown_chunk.extend_from_slice(&unknown_chunk);
        png_with_unknown_chunk.extend_from_slice(&png_chunk(b"IEND", &[]));
        let png_with_unknown_document = MetadataDocument::parse(
            &png_with_unknown_chunk,
            Some("unknown.png"),
            None,
            MetadataLimits::default(),
        )
        .expect("PNG with an unknown ancillary chunk");
        let png_with_unknown_output = png_with_unknown_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("PNG unknown-chunk-preserving write");
        cases.insert(
            "png_unknown_chunk_preserved".to_owned(),
            png_with_unknown_output
                .windows(unknown_chunk.len())
                .any(|window| window == unknown_chunk),
        );

        let safetensors_header = br#"{"__metadata__":{"future":"keep"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut safetensors = Vec::new();
        safetensors.extend_from_slice(&(safetensors_header.len() as u64).to_le_bytes());
        safetensors.extend_from_slice(safetensors_header);
        safetensors.extend_from_slice(&[1, 2, 3, 4]);
        let safetensors_document = MetadataDocument::parse(
            &safetensors,
            Some("fixture.safetensors"),
            None,
            MetadataLimits::default(),
        )
        .expect("safetensors fixture");
        let safetensors_output = safetensors_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("safetensors metadata write");
        let safetensors_reopened = MetadataDocument::parse(
            &safetensors_output,
            Some("fixture.safetensors"),
            None,
            MetadataLimits::default(),
        )
        .expect("safetensors metadata reopen");
        cases.insert(
            "safetensors_preserves_tensor_and_unknown_metadata".to_owned(),
            safetensors_output.ends_with(&[1, 2, 3, 4])
                && safetensors_reopened.get_case_insensitive("future") == Some("keep")
                && safetensors_reopened.get_case_insensitive("workflow")
                    == fields().get("workflow").map(String::as_str),
        );

        let mut glb = glb_fixture();
        glb.extend_from_slice(&4u32.to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&[1, 2, 3, 4]);
        let glb_length = u32::try_from(glb.len()).expect("GLB fixture length");
        glb[8..12].copy_from_slice(&glb_length.to_le_bytes());
        let glb_json_length = usize::try_from(u32::from_le_bytes(
            glb[12..16].try_into().expect("GLB JSON length"),
        ))
        .expect("GLB JSON length fits");
        let glb_binary = glb[20 + glb_json_length..].to_vec();
        let glb_document =
            MetadataDocument::parse(&glb, Some("fixture.glb"), None, MetadataLimits::default())
                .expect("GLB fixture");
        let glb_output = glb_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("GLB metadata write");
        let glb_reopened = MetadataDocument::parse(
            &glb_output,
            Some("fixture.glb"),
            None,
            MetadataLimits::default(),
        )
        .expect("GLB metadata reopen");
        cases.insert(
            "glb_write_preserves_binary".to_owned(),
            glb_output.ends_with(&glb_binary)
                && glb_reopened.get_case_insensitive("workflow")
                    == fields().get("workflow").map(String::as_str),
        );

        let svg = br#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#;
        let svg_document =
            MetadataDocument::parse(svg, Some("fixture.svg"), None, MetadataLimits::default())
                .expect("SVG fixture");
        let svg_output = svg_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy {
                    metadata_enabled: false,
                },
                &MetadataLimits::default(),
            )
            .expect("SVG disabled-metadata write");
        let png_disabled_output = png_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy {
                    metadata_enabled: false,
                },
                &MetadataLimits::default(),
            )
            .expect("PNG disabled-metadata write");
        let malformed = MetadataDocument::parse(
            b"\x89PNG\r\n\x1a\n\0",
            None,
            Some("image/png"),
            MetadataLimits::default(),
        )
        .expect("malformed input remains inspectable");
        cases.insert(
            "malformed_visible".to_owned(),
            malformed.entries().is_empty() && !malformed.diagnostics().is_empty(),
        );
        let small_limits = MetadataLimits {
            max_input_bytes: 4,
            ..MetadataLimits::default()
        };
        cases.insert(
            "oversized_rejected".to_owned(),
            matches!(
                MetadataDocument::parse(b"12345", None, None, small_limits),
                Err(MetadataError::InputTooLarge { .. })
            ),
        );
        cases.insert(
            "content_precedes_extension".to_owned(),
            detect_carrier(&png, Some("wrong.mp4"), Some("video/mp4")) == MetadataCarrier::Png,
        );
        let isobmff_variants = isobmff_fixture();
        let mov_document = MetadataDocument::parse(
            &isobmff_variants,
            Some("fixture.MOV"),
            Some("video/quicktime"),
            MetadataLimits::default(),
        )
        .expect("MOV metadata fixture");
        let m4v_document = MetadataDocument::parse(
            &isobmff_variants,
            Some("fixture.M4V"),
            Some("video/x-m4v"),
            MetadataLimits::default(),
        )
        .expect("M4V metadata fixture");
        cases.insert(
            "isobmff_mp4_mov_m4v_adapter".to_owned(),
            [mov_document, m4v_document].iter().all(|document| {
                document.carrier() == MetadataCarrier::IsobmffVideo
                    && document.get_case_insensitive("workflow").is_some()
                    && document.original_bytes() == isobmff_variants
            }),
        );
        let mut boundary_mp3 = b"ID3".to_vec();
        boundary_mp3.resize(4_090, 0);
        boundary_mp3
            .extend_from_slice(b"prompt\0{\"1\":{\"class_type\":\"KSampler\",\"inputs\":{}}}\0");
        boundary_mp3
            .extend_from_slice(b"workflow\0{\"version\":0.4,\"nodes\":[],\"links\":[]}\0\xff\xfb");
        let boundary_mp3_document = MetadataDocument::parse(
            &boundary_mp3,
            Some("boundary.mp3"),
            Some("audio/mpeg"),
            MetadataLimits::default(),
        )
        .expect("MP3 page-boundary fixture");
        cases.insert(
            "mp3_metadata_crosses_source_page_boundary".to_owned(),
            boundary_mp3_document
                .get_case_insensitive("prompt")
                .is_some()
                && boundary_mp3_document
                    .get_case_insensitive("workflow")
                    .is_some()
                && boundary_mp3_document.original_bytes() == boundary_mp3,
        );
        let invalid_signature_mp3 =
            b"BADprompt\0{\"1\":{\"class_type\":\"KSampler\",\"inputs\":{}}}\0";
        let invalid_signature_mp3_document = MetadataDocument::parse(
            invalid_signature_mp3,
            Some("invalid-signature.mp3"),
            Some("audio/mpeg"),
            MetadataLimits::default(),
        )
        .expect("invalid-signature MP3 fallback");
        cases.insert(
            "mp3_invalid_signature_fallback_is_visible".to_owned(),
            invalid_signature_mp3_document
                .get_case_insensitive("prompt")
                .is_some()
                && invalid_signature_mp3_document.diagnostics().len() == 1
                && invalid_signature_mp3_document.original_bytes() == invalid_signature_mp3,
        );
        let invalid_signature_ogg =
            b"BADprompt={\"1\":{\"class_type\":\"KSampler\",\"inputs\":{}}}\0";
        let invalid_signature_ogg_document = MetadataDocument::parse(
            invalid_signature_ogg,
            Some("invalid-signature.ogg"),
            Some("audio/ogg"),
            MetadataLimits::default(),
        )
        .expect("invalid-signature Ogg fallback");
        cases.insert(
            "ogg_invalid_signature_fallback_is_visible".to_owned(),
            invalid_signature_ogg_document
                .get_case_insensitive("prompt")
                .is_some()
                && invalid_signature_ogg_document.diagnostics().len() == 1
                && invalid_signature_ogg_document.original_bytes() == invalid_signature_ogg,
        );
        let ply = MetadataDocument::parse(
            b"ply\nformat ascii 1.0\nend_header\n",
            Some("fixture.ply"),
            None,
            MetadataLimits::default(),
        )
        .expect("PLY fixture");
        cases.insert(
            "ply_explicit_noncarrier".to_owned(),
            ply.support() == MetadataSupport::ExplicitNonCarrier && ply.entries().is_empty(),
        );

        let (exr, old_exr_offset) = open_exr_fixture();
        let exr_document =
            MetadataDocument::parse(&exr, Some("fixture.exr"), None, MetadataLimits::default())
                .expect("EXR fixture");
        let exr_output = exr_document
            .embed_sim_metadata(
                &fields(),
                MetadataWritePolicy::default(),
                &MetadataLimits::default(),
            )
            .expect("EXR metadata write");
        let exr_table_position = exr_output.len() - b"pixels".len() - 8;
        let shifted_exr_offset = u64::from_le_bytes(
            exr_output[exr_table_position..exr_table_position + 8]
                .try_into()
                .expect("EXR chunk offset"),
        );
        cases.insert(
            "exr_absolute_offsets_shifted".to_owned(),
            exr_document.support() == MetadataSupport::WriteOnly
                && shifted_exr_offset == old_exr_offset + (exr_output.len() - exr.len()) as u64
                && exr_output.ends_with(b"pixels"),
        );
        let disabled_policy = MetadataWritePolicy {
            metadata_enabled: false,
        };
        let glb_disabled_output = glb_document
            .embed_sim_metadata(&fields(), disabled_policy, &MetadataLimits::default())
            .expect("suppressed GLB metadata");
        let safetensors_disabled_output = safetensors_document
            .embed_sim_metadata(&fields(), disabled_policy, &MetadataLimits::default())
            .expect("suppressed safetensors metadata");
        let exr_disabled_output = exr_document
            .embed_sim_metadata(&fields(), disabled_policy, &MetadataLimits::default())
            .expect("suppressed EXR metadata");
        cases.insert(
            "stage_writer_disable_metadata_matrix".to_owned(),
            svg_output != svg
                && png_disabled_output == png
                && glb_disabled_output == glb
                && safetensors_disabled_output == safetensors
                && exr_disabled_output == exr,
        );

        let hostile_fixtures = [
            compressed_itxt_png(),
            png_with_bad_itxt_then_prompt(),
            webp_fixture(),
            avif_fixture(),
            flac_fixture(),
            ogg_fixture(),
            webm_fixture(),
            isobmff_fixture(),
            glb_fixture(),
        ];
        let hostile_truncations = hostile_fixtures
            .iter()
            .map(Vec::len)
            .map(|length| length + 1)
            .sum::<usize>();
        let hostile_preserved = hostile_fixtures.iter().all(|fixture| {
            (0..=fixture.len()).all(|length| {
                let truncated = &fixture[..length];
                MetadataDocument::parse(truncated, None, None, MetadataLimits::default())
                    .is_ok_and(|document| document.original_bytes() == truncated)
            })
        });
        cases.insert(
            "hostile_truncations_preserved".to_owned(),
            hostile_preserved,
        );
        let unknown_length_webm = [
            0x1a, 0x45, 0xdf, 0xa3, 0x80, 0x67, 0xc8, 0xff, 0x45, 0xa3, 0x81, b'x',
        ];
        let unknown_length_document = MetadataDocument::parse(
            unknown_length_webm,
            Some("unknown-length.webm"),
            None,
            MetadataLimits::default(),
        )
        .expect("unknown-length WebM remains inspectable");
        cases.insert(
            "webm_unknown_length_tag_rejected".to_owned(),
            unknown_length_document.entries().is_empty()
                && unknown_length_document.original_bytes() == unknown_length_webm,
        );

        assert!(cases.values().all(|passed| *passed), "{cases:?}");
        let mut fixture_digests = carrier_digests()
            .into_iter()
            .map(|(name, digest)| (name.to_owned(), digest))
            .collect::<BTreeMap<_, _>>();
        fixture_digests.insert(
            "safetensors".to_owned(),
            format!("{:x}", Sha256::digest(&safetensors)),
        );
        fixture_digests.insert(
            "png-write".to_owned(),
            format!("{:x}", Sha256::digest(&png)),
        );
        fixture_digests.insert(
            "exr-write".to_owned(),
            format!("{:x}", Sha256::digest(&exr)),
        );
        fixture_digests.insert(
            "svg-nonfinite".to_owned(),
            format!("{:x}", Sha256::digest(nonfinite_svg)),
        );
        fixture_digests.insert(
            "glb-nonfinite".to_owned(),
            format!("{:x}", Sha256::digest(&nonfinite_glb)),
        );
        fixture_digests.insert(
            "mp3-page-boundary".to_owned(),
            format!("{:x}", Sha256::digest(&boundary_mp3)),
        );
        fixture_digests.insert(
            "isobmff-variants".to_owned(),
            format!("{:x}", Sha256::digest(&isobmff_variants)),
        );
        fixture_digests.insert(
            "avif-usercomment".to_owned(),
            format!("{:x}", Sha256::digest(&avif_usercomment)),
        );
        fixture_digests.insert(
            "png-bad-itxt".to_owned(),
            format!("{:x}", Sha256::digest(&bad_itxt_png)),
        );
        fixture_digests.insert(
            "mp3-invalid-signature".to_owned(),
            format!("{:x}", Sha256::digest(invalid_signature_mp3)),
        );
        fixture_digests.insert(
            "ogg-invalid-signature".to_owned(),
            format!("{:x}", Sha256::digest(invalid_signature_ogg)),
        );
        let artifact = json!({
            "validation": "VAL-METADATA-001",
            "scope": "embedded-metadata-carriers",
            "environment": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "backend": "native-rust"},
            "cases": cases,
            "fixture_sha256": fixture_digests,
            "source_fixture_sha256": source_fixture_digests,
            "source_contract": "projects/comfy/ComfyUI-Frontend/src/scripts/metadata",
            "delegated_contracts": {"direct-json": "sim_runtime::workflow_formats"},
            "hostile_truncation_count": hostile_truncations,
            "skipped": [],
            "subprocesses": 0,
        });
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("target")
            });
        let directory = target.join("sim-parity");
        fs::create_dir_all(&directory).expect("artifact directory");
        fs::write(
            directory.join("val-metadata-001.json"),
            serde_json::to_vec_pretty(&artifact).expect("artifact JSON"),
        )
        .expect("artifact write");
    }

    #[test]
    fn deterministic_hostile_truncation_matrix_never_panics_or_changes_source() {
        let fixtures = [
            compressed_itxt_png(),
            png_with_bad_itxt_then_prompt(),
            webp_fixture(),
            avif_fixture(),
            flac_fixture(),
            ogg_fixture(),
            webm_fixture(),
            isobmff_fixture(),
            glb_fixture(),
        ];
        for fixture in fixtures {
            for length in 0..=fixture.len() {
                let truncated = &fixture[..length];
                let document =
                    MetadataDocument::parse(truncated, None, None, MetadataLimits::default())
                        .expect("bounded hostile fixture");
                assert_eq!(document.original_bytes(), truncated);
            }
        }
    }
}
