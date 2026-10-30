use anyhow::{Context as _, Result, anyhow, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const CERTIFICATION_STATUS: &str = "pending_real_ffmpeg_7_1_x86_64_linux";
const STRUCTURAL_MARKER: &[u8] = b"zed-sim-general-demux-decode-structural-v1\0";
const GENERATOR_SOURCE: &[u8] = include_bytes!("generate_video_general_demux_decode_fixture.rs");

const SOURCE_PINS: [(&str, &str); 6] = [
    (
        "projects/comfy/ComfyUI/comfy_api/latest/_input_impl/video_types.py",
        "b74216bc76178e9a21769e60cf4fd910183a19236cfa7fb352e74f2b13e482e2",
    ),
    (
        "projects/comfy/ComfyUI/comfy_api/latest/_input/video_types.py",
        "6cba96f1f254094436de6c3e64c0a7b469e7cbaf3578c234c18eea88cf93926d",
    ),
    (
        "projects/comfy/ComfyUI/comfy_api/latest/_input/basic_types.py",
        "99251ecb404c881f28a413d4c0829fdf5345a1897dfac76ed6199863601098b1",
    ),
    (
        "projects/comfy/ComfyUI/comfy_api/latest/_util/video_types.py",
        "5d2f7dd2b82aecec3c05f8f5bd9519530736825d33590207ea0aa32b8cc8627f",
    ),
    (
        "projects/comfy/ComfyUI/requirements.txt",
        "48f4835af39b753fb2e637ec17813716024e08952e82e6e4e536a0fcfd944d0e",
    ),
    (
        "crates/sim_test_support/parity/baseline.md",
        "7779e10ec20426f6c5e4e23e22290ee6bc6776d70e9cdc1712c7c8887e116cdd",
    ),
];

#[derive(Clone, Serialize)]
struct FixtureCase {
    id: &'static str,
    path: String,
    container: &'static str,
    role: &'static str,
    byte_length: usize,
    sha256: String,
    video_streams: Vec<StreamDisposition>,
    audio_streams: Vec<StreamDisposition>,
    expected_disposition: &'static str,
}

#[derive(Clone, Serialize)]
struct StreamDisposition {
    ordinal: u8,
    codec: &'static str,
    decodable: bool,
    selected: bool,
    diagnostic: Option<&'static str>,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u16,
    fixture_class: &'static str,
    certification_status: &'static str,
    certified_interoperability: bool,
    cases: Vec<FixtureCase>,
    source_behaviors: Vec<&'static str>,
}

#[derive(Serialize)]
struct NumericOracle {
    schema_version: u16,
    certification_status: &'static str,
    decoded_output_hashes_observed: bool,
    cases: Vec<NumericCase>,
}

#[derive(Serialize)]
struct NumericCase {
    id: &'static str,
    source_width: u16,
    source_height: u16,
    aligned_width: u16,
    aligned_height: u16,
    output_width: u16,
    output_height: u16,
    frame_rate_numerator: u32,
    frame_rate_denominator: u32,
    frame_count: u16,
    bit_depth: u8,
    alpha: bool,
    rotation_quarter_turns: i8,
    audio_channels: u8,
    audio_sample_rate: u32,
    trimmed_audio_samples_per_channel: u32,
    enters_width_alignment_branch: bool,
    decoded_video_sha256: Option<String>,
    decoded_alpha_sha256: Option<String>,
    decoded_audio_sha256: Option<String>,
}

#[derive(Serialize)]
struct Provenance {
    schema_version: u16,
    generator_sha256: String,
    fixture_manifest_sha256: String,
    numeric_oracle_sha256: String,
    input_coverage_sha256: String,
    certification_status: &'static str,
    certified_interoperability: bool,
    host_codec_used: bool,
    compiler_used: bool,
    subprocess_used: bool,
    network_used: bool,
    credential_used: bool,
    sources: Vec<SourcePin>,
}

#[derive(Serialize)]
struct SourcePin {
    path: &'static str,
    sha256: &'static str,
}

fn main() -> Result<()> {
    let check = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(argument) => bail!("unsupported argument {argument}"),
    };
    let fixture = build_fixture()?;
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/video/general-demux-decode");
    if check {
        check_fixture(&root, &fixture)
    } else {
        write_fixture(&root, &fixture)
    }
}

fn build_fixture() -> Result<BTreeMap<String, Vec<u8>>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("workspace root is unavailable"))?;
    let sources = verify_sources(workspace)?;
    let mut files = BTreeMap::new();
    let mut cases = Vec::new();

    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "h264-8bit-aac-rotation",
            filename: "h264-8bit-aac-rotation.mp4",
            container: "mp4",
            role: "first-video-last-decodable-audio-rotation",
            bytes: mp4_fixture("h264-8bit-aac-rotation", true)?,
            video_streams: vec![
                video_stream(0, "h264", true, true, None),
                video_stream(1, "h264", true, false, None),
            ],
            audio_streams: vec![
                audio_stream(0, "aac", true, false, None),
                audio_stream(
                    1,
                    "apac",
                    false,
                    false,
                    Some("unsupported_audio_stream_skipped"),
                ),
                audio_stream(2, "aac", true, true, None),
            ],
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "h264-10bit",
            filename: "h264-10bit.mp4",
            container: "mp4",
            role: "ten-bit-depth",
            bytes: mp4_fixture("h264-10bit", true)?,
            video_streams: vec![video_stream(0, "h264", true, true, None)],
            audio_streams: Vec::new(),
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "vp9-alpha",
            filename: "vp9-alpha.webm",
            container: "webm",
            role: "alpha-detection",
            bytes: webm_fixture("vp9-alpha"),
            video_streams: vec![video_stream(0, "vp9", true, true, None)],
            audio_streams: Vec::new(),
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "av1-average-rate-fallback",
            filename: "av1-average-rate-fallback.webm",
            container: "webm",
            role: "av1-one-fps-fallback",
            bytes: webm_fixture("av1-average-rate-fallback"),
            video_streams: vec![video_stream(0, "av1", true, true, None)],
            audio_streams: Vec::new(),
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "width-non-32-aligned",
            filename: "width-non-32-aligned.mp4",
            container: "mp4",
            role: "pad-smear-crop-even-width",
            bytes: mp4_fixture("width-non-32-aligned", true)?,
            video_streams: vec![video_stream(0, "h264", true, true, None)],
            audio_streams: Vec::new(),
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "height-only-nonalignment",
            filename: "height-only-nonalignment.mp4",
            container: "mp4",
            role: "height-only-does-not-align",
            bytes: mp4_fixture("height-only-nonalignment", true)?,
            video_streams: vec![video_stream(0, "h264", true, true, None)],
            audio_streams: Vec::new(),
            expected_disposition: "decode_after_certification",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "missing-video-stream",
            filename: "missing-video-stream.mp4",
            container: "mp4",
            role: "missing-video-stream",
            bytes: mp4_fixture("missing-video-stream", false)?,
            video_streams: Vec::new(),
            audio_streams: vec![audio_stream(0, "aac", true, true, None)],
            expected_disposition: "missing_video_stream",
        },
    );
    let truncated = mp4_fixture("truncated", true)?;
    let truncated = truncated
        .get(..19)
        .ok_or_else(|| anyhow!("truncated fixture source is too short"))?
        .to_vec();
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "truncated",
            filename: "truncated.mp4",
            container: "mp4",
            role: "truncated-container",
            bytes: truncated,
            video_streams: Vec::new(),
            audio_streams: Vec::new(),
            expected_disposition: "invalid_data",
        },
    );
    push_case(
        &mut files,
        &mut cases,
        CaseInput {
            id: "malformed",
            filename: "malformed.bin",
            container: "unknown",
            role: "malformed-container",
            bytes: b"not-a-media-container\n".to_vec(),
            video_streams: Vec::new(),
            audio_streams: Vec::new(),
            expected_disposition: "invalid_data",
        },
    );

    let manifest = Manifest {
        schema_version: 1,
        fixture_class: "deterministic_offline_structural_input_seed",
        certification_status: CERTIFICATION_STATUS,
        certified_interoperability: false,
        cases,
        source_behaviors: vec![
            "first_video_stream",
            "last_decodable_audio_stream_with_skipped_unsupported_diagnostic",
            "average_frame_rate_then_one_fps_fallback",
            "bit_depth_from_maximum_component_bits",
            "alpha_from_component_or_pal8",
            "non_yuvj_width_alignment_pad_then_smear_then_crop",
            "height_only_nonalignment_does_not_enter_alignment_branch",
            "quarter_turn_rotation_after_conversion",
            "float_planar_audio_resample_then_trim",
        ],
    };
    let manifest_bytes = canonical_json(&manifest)?;
    let oracle_bytes = canonical_json(&numeric_oracle())?;
    let provenance = Provenance {
        schema_version: 1,
        generator_sha256: sha256(GENERATOR_SOURCE),
        fixture_manifest_sha256: sha256(&manifest_bytes),
        numeric_oracle_sha256: sha256(&oracle_bytes),
        input_coverage_sha256: sha256(&coverage_bytes(&files)),
        certification_status: CERTIFICATION_STATUS,
        certified_interoperability: false,
        host_codec_used: false,
        compiler_used: false,
        subprocess_used: false,
        network_used: false,
        credential_used: false,
        sources,
    };
    files.insert("manifest.json".to_owned(), manifest_bytes);
    files.insert("oracle.json".to_owned(), oracle_bytes);
    files.insert("provenance.json".to_owned(), canonical_json(&provenance)?);
    Ok(files)
}

struct CaseInput {
    id: &'static str,
    filename: &'static str,
    container: &'static str,
    role: &'static str,
    bytes: Vec<u8>,
    video_streams: Vec<StreamDisposition>,
    audio_streams: Vec<StreamDisposition>,
    expected_disposition: &'static str,
}

fn push_case(
    files: &mut BTreeMap<String, Vec<u8>>,
    cases: &mut Vec<FixtureCase>,
    input: CaseInput,
) {
    let path = format!("inputs/{}", input.filename);
    cases.push(FixtureCase {
        id: input.id,
        path: path.clone(),
        container: input.container,
        role: input.role,
        byte_length: input.bytes.len(),
        sha256: sha256(&input.bytes),
        video_streams: input.video_streams,
        audio_streams: input.audio_streams,
        expected_disposition: input.expected_disposition,
    });
    files.insert(path, input.bytes);
}

fn video_stream(
    ordinal: u8,
    codec: &'static str,
    decodable: bool,
    selected: bool,
    diagnostic: Option<&'static str>,
) -> StreamDisposition {
    StreamDisposition {
        ordinal,
        codec,
        decodable,
        selected,
        diagnostic,
    }
}

fn audio_stream(
    ordinal: u8,
    codec: &'static str,
    decodable: bool,
    selected: bool,
    diagnostic: Option<&'static str>,
) -> StreamDisposition {
    StreamDisposition {
        ordinal,
        codec,
        decodable,
        selected,
        diagnostic,
    }
}

fn numeric_oracle() -> NumericOracle {
    NumericOracle {
        schema_version: 1,
        certification_status: CERTIFICATION_STATUS,
        decoded_output_hashes_observed: false,
        cases: vec![
            numeric_case(
                "h264-8bit-aac-rotation",
                [34, 18, 64, 32, 18, 34],
                [24, 1],
                3,
                8,
                false,
                1,
                (2, 48_000, 6_000),
                true,
            ),
            numeric_case(
                "h264-10bit",
                [32, 16, 32, 16, 32, 16],
                [30_000, 1_001],
                2,
                10,
                false,
                0,
                (0, 0, 0),
                false,
            ),
            numeric_case(
                "vp9-alpha",
                [18, 10, 32, 32, 18, 10],
                [12, 1],
                2,
                8,
                true,
                0,
                (0, 0, 0),
                true,
            ),
            numeric_case(
                "av1-average-rate-fallback",
                [32, 10, 32, 10, 32, 10],
                [1, 1],
                2,
                10,
                false,
                0,
                (0, 0, 0),
                false,
            ),
            numeric_case(
                "width-non-32-aligned",
                [34, 32, 64, 32, 34, 32],
                [25, 1],
                2,
                8,
                false,
                0,
                (0, 0, 0),
                true,
            ),
            numeric_case(
                "height-only-nonalignment",
                [32, 18, 32, 18, 32, 18],
                [25, 1],
                2,
                8,
                false,
                0,
                (0, 0, 0),
                false,
            ),
        ],
    }
}

#[allow(clippy::too_many_arguments)]
fn numeric_case(
    id: &'static str,
    dimensions: [u16; 6],
    frame_rate: [u32; 2],
    frame_count: u16,
    bit_depth: u8,
    alpha: bool,
    rotation_quarter_turns: i8,
    audio: (u8, u32, u32),
    enters_width_alignment_branch: bool,
) -> NumericCase {
    NumericCase {
        id,
        source_width: dimensions[0],
        source_height: dimensions[1],
        aligned_width: dimensions[2],
        aligned_height: dimensions[3],
        output_width: dimensions[4],
        output_height: dimensions[5],
        frame_rate_numerator: frame_rate[0],
        frame_rate_denominator: frame_rate[1],
        frame_count,
        bit_depth,
        alpha,
        rotation_quarter_turns,
        audio_channels: audio.0,
        audio_sample_rate: audio.1,
        trimmed_audio_samples_per_channel: audio.2,
        enters_width_alignment_branch,
        decoded_video_sha256: None,
        decoded_alpha_sha256: None,
        decoded_audio_sha256: None,
    }
}

fn verify_sources(workspace: &Path) -> Result<Vec<SourcePin>> {
    SOURCE_PINS
        .iter()
        .map(|(path, expected_sha256)| {
            let bytes = fs::read(workspace.join(path))
                .with_context(|| format!("read pinned source {path}"))?;
            let actual_sha256 = sha256(&bytes);
            if actual_sha256 != *expected_sha256 {
                bail!("pinned source digest changed for {path}")
            }
            Ok(SourcePin {
                path,
                sha256: expected_sha256,
            })
        })
        .collect()
}

fn mp4_fixture(id: &str, has_video_payload: bool) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&mp4_box(*b"ftyp", b"isom\0\0\x02\0isomiso2")?);
    let mut disposition = STRUCTURAL_MARKER.to_vec();
    disposition.extend_from_slice(id.as_bytes());
    bytes.extend_from_slice(&mp4_box(*b"uuid", &disposition)?);
    if has_video_payload {
        bytes.extend_from_slice(&mp4_box(*b"mdat", id.as_bytes())?);
    }
    Ok(bytes)
}

fn mp4_box(kind: [u8; 4], payload: &[u8]) -> Result<Vec<u8>> {
    let byte_length = payload
        .len()
        .checked_add(8)
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| anyhow!("MP4 fixture box length overflowed"))?;
    let mut bytes = Vec::with_capacity(byte_length as usize);
    bytes.extend_from_slice(&byte_length.to_be_bytes());
    bytes.extend_from_slice(&kind);
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

fn webm_fixture(id: &str) -> Vec<u8> {
    let mut bytes = vec![
        0x1a, 0x45, 0xdf, 0xa3, 0x9f, 0x42, 0x86, 0x81, 0x01, 0x42, 0xf7, 0x81, 0x01, 0x42, 0xf2,
        0x81, 0x04, 0x42, 0xf3, 0x81, 0x08, 0x42, 0x82, 0x84, b'w', b'e', b'b', b'm', 0x42, 0x87,
        0x81, 0x04, 0x42, 0x85, 0x81, 0x02,
    ];
    bytes.extend_from_slice(STRUCTURAL_MARKER);
    bytes.extend_from_slice(id.as_bytes());
    bytes
}

fn canonical_json(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn coverage_bytes(files: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut coverage = Vec::new();
    for (path, bytes) in files {
        coverage
            .extend_from_slice(format!("{} {}  {path}\n", sha256(bytes), bytes.len()).as_bytes());
    }
    coverage
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_fixture(root: &Path, fixture: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    for (path, bytes) in fixture {
        let destination = root.join(path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(destination, bytes)?;
    }
    check_fixture(root, fixture)
}

fn check_fixture(root: &Path, fixture: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    let actual_paths = recursive_files(root)?;
    let expected_paths = fixture.keys().cloned().collect::<BTreeSet<_>>();
    if actual_paths != expected_paths {
        bail!("tracked fixture file set differs from deterministic output")
    }
    for (path, expected) in fixture {
        let actual = fs::read(root.join(path)).with_context(|| format!("read fixture {path}"))?;
        if &actual != expected {
            bail!("tracked fixture {path} differs from deterministic output")
        }
    }
    Ok(())
}

fn recursive_files(root: &Path) -> Result<BTreeSet<String>> {
    fn visit(root: &Path, current: &Path, output: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(current)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                visit(root, &path, output)?;
            } else if file_type.is_file() {
                let relative = path
                    .strip_prefix(root)?
                    .to_str()
                    .ok_or_else(|| anyhow!("fixture path is not UTF-8"))?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                output.insert(relative);
            } else {
                bail!("fixture contains a non-regular entry")
            }
        }
        Ok(())
    }

    let mut output = BTreeSet::new();
    if root.exists() {
        visit(root, root, &mut output)?;
    }
    Ok(output)
}
