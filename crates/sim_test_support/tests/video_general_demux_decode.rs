use anyhow::{Context as _, Result, anyhow, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const CERTIFICATION_STATUS: &str = "pending_real_ffmpeg_7_1_x86_64_linux";
const GENERATOR_SOURCE: &[u8] =
    include_bytes!("../src/bin/generate_video_general_demux_decode_fixture.rs");

#[derive(Deserialize)]
struct Manifest {
    schema_version: u16,
    fixture_class: String,
    certification_status: String,
    certified_interoperability: bool,
    cases: Vec<FixtureCase>,
    source_behaviors: Vec<String>,
}

#[derive(Deserialize)]
struct FixtureCase {
    id: String,
    path: String,
    container: String,
    role: String,
    byte_length: usize,
    sha256: String,
    video_streams: Vec<StreamDisposition>,
    audio_streams: Vec<StreamDisposition>,
    expected_disposition: String,
}

#[derive(Deserialize)]
struct StreamDisposition {
    ordinal: u8,
    codec: String,
    decodable: bool,
    selected: bool,
    diagnostic: Option<String>,
}

#[derive(Deserialize)]
struct NumericOracle {
    schema_version: u16,
    certification_status: String,
    decoded_output_hashes_observed: bool,
    cases: Vec<NumericCase>,
}

#[derive(Deserialize)]
struct NumericCase {
    id: String,
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

#[derive(Deserialize)]
struct Provenance {
    schema_version: u16,
    generator_sha256: String,
    fixture_manifest_sha256: String,
    numeric_oracle_sha256: String,
    input_coverage_sha256: String,
    certification_status: String,
    certified_interoperability: bool,
    host_codec_used: bool,
    compiler_used: bool,
    subprocess_used: bool,
    network_used: bool,
    credential_used: bool,
    sources: Vec<SourcePin>,
}

#[derive(Deserialize)]
struct SourcePin {
    path: String,
    sha256: String,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/video/general-demux-decode")
}

fn workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow!("workspace root is unavailable"))
}

#[test]
fn general_demux_decode_fixture_is_exact_and_explicitly_uncertified() -> Result<()> {
    let root = fixture_root();
    let manifest_bytes = fs::read(root.join("manifest.json"))?;
    let oracle_bytes = fs::read(root.join("oracle.json"))?;
    let provenance_bytes = fs::read(root.join("provenance.json"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    let oracle: NumericOracle = serde_json::from_slice(&oracle_bytes)?;
    let provenance: Provenance = serde_json::from_slice(&provenance_bytes)?;

    assert_eq!(manifest.schema_version, 1);
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(provenance.schema_version, 1);
    assert_eq!(manifest.certification_status, CERTIFICATION_STATUS);
    assert_eq!(oracle.certification_status, CERTIFICATION_STATUS);
    assert_eq!(provenance.certification_status, CERTIFICATION_STATUS);
    assert_eq!(
        manifest.fixture_class,
        "deterministic_offline_structural_input_seed"
    );
    assert!(!manifest.certified_interoperability);
    assert!(!provenance.certified_interoperability);
    assert!(!oracle.decoded_output_hashes_observed);
    assert!(!provenance.host_codec_used);
    assert!(!provenance.compiler_used);
    assert!(!provenance.subprocess_used);
    assert!(!provenance.network_used);
    assert!(!provenance.credential_used);
    assert_eq!(provenance.generator_sha256, sha256(GENERATOR_SOURCE));
    assert_eq!(provenance.fixture_manifest_sha256, sha256(&manifest_bytes));
    assert_eq!(provenance.numeric_oracle_sha256, sha256(&oracle_bytes));

    let expected_ids = BTreeSet::from([
        "av1-average-rate-fallback",
        "h264-10bit",
        "h264-8bit-aac-rotation",
        "height-only-nonalignment",
        "malformed",
        "missing-video-stream",
        "truncated",
        "vp9-alpha",
        "width-non-32-aligned",
    ]);
    let actual_ids = manifest
        .cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_ids, expected_ids);

    let expected_paths = manifest
        .cases
        .iter()
        .map(|case| case.path.clone())
        .chain(["manifest.json", "oracle.json", "provenance.json"].map(str::to_owned))
        .collect::<BTreeSet<_>>();
    assert_eq!(recursive_files(&root)?, expected_paths);
    let mut input_files = BTreeMap::new();
    for case in &manifest.cases {
        let bytes = fs::read(root.join(&case.path))
            .with_context(|| format!("read fixture {}", case.path))?;
        assert_eq!(bytes.len(), case.byte_length, "{}", case.id);
        assert_eq!(sha256(&bytes), case.sha256, "{}", case.id);
        assert!(!case.role.is_empty(), "{}", case.id);
        match case.container.as_str() {
            "mp4" if case.id == "truncated" => {
                assert!(bytes.len() < 24);
                assert_eq!(bytes.get(4..8), Some(b"ftyp".as_slice()));
            }
            "mp4" => assert_mp4_structural_seed(&bytes, &case.id)?,
            "webm" => assert_webm_structural_seed(&bytes, &case.id)?,
            "unknown" => {
                assert_eq!(case.id, "malformed");
                assert!(!bytes.starts_with(b"\x1a\x45\xdf\xa3"));
                assert_ne!(bytes.get(4..8), Some(b"ftyp".as_slice()));
            }
            other => bail!("unexpected fixture container {other}"),
        }
        input_files.insert(case.path.clone(), bytes);
    }
    assert_eq!(
        provenance.input_coverage_sha256,
        sha256(&coverage_bytes(&input_files))
    );
    Ok(())
}

#[test]
fn general_demux_decode_fixture_pins_source_selection_and_numeric_boundaries() -> Result<()> {
    let root = fixture_root();
    let manifest: Manifest = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let oracle: NumericOracle = serde_json::from_slice(&fs::read(root.join("oracle.json"))?)?;
    let provenance: Provenance = serde_json::from_slice(&fs::read(root.join("provenance.json"))?)?;

    let expected_behaviors = [
        "first_video_stream",
        "last_decodable_audio_stream_with_skipped_unsupported_diagnostic",
        "average_frame_rate_then_one_fps_fallback",
        "bit_depth_from_maximum_component_bits",
        "alpha_from_component_or_pal8",
        "non_yuvj_width_alignment_pad_then_smear_then_crop",
        "height_only_nonalignment_does_not_enter_alignment_branch",
        "quarter_turn_rotation_after_conversion",
        "float_planar_audio_resample_then_trim",
    ];
    assert_eq!(
        manifest.source_behaviors,
        expected_behaviors.map(str::to_owned)
    );

    let selected = case(&manifest, "h264-8bit-aac-rotation")?;
    assert_eq!(selected.video_streams.len(), 2);
    assert_stream(&selected.video_streams[0], 0, "h264", true, true, None);
    assert_stream(&selected.video_streams[1], 1, "h264", true, false, None);
    assert_eq!(selected.audio_streams.len(), 3);
    assert_stream(&selected.audio_streams[0], 0, "aac", true, false, None);
    assert_stream(
        &selected.audio_streams[1],
        1,
        "apac",
        false,
        false,
        Some("unsupported_audio_stream_skipped"),
    );
    assert_stream(&selected.audio_streams[2], 2, "aac", true, true, None);
    assert_eq!(selected.expected_disposition, "decode_after_certification");

    let missing = case(&manifest, "missing-video-stream")?;
    assert!(missing.video_streams.is_empty());
    assert_eq!(missing.expected_disposition, "missing_video_stream");
    for failure in ["truncated", "malformed"] {
        assert_eq!(
            case(&manifest, failure)?.expected_disposition,
            "invalid_data"
        );
    }

    let numerics = oracle
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    assert_numeric_case(
        required_numeric(&numerics, "h264-8bit-aac-rotation")?,
        [34, 18, 64, 32, 18, 34],
        [24, 1],
        [3, 8, 0, 1],
        [2, 48_000, 6_000],
        true,
    );
    assert_numeric_case(
        required_numeric(&numerics, "h264-10bit")?,
        [32, 16, 32, 16, 32, 16],
        [30_000, 1_001],
        [2, 10, 0, 0],
        [0, 0, 0],
        false,
    );
    assert_numeric_case(
        required_numeric(&numerics, "vp9-alpha")?,
        [18, 10, 32, 32, 18, 10],
        [12, 1],
        [2, 8, 1, 0],
        [0, 0, 0],
        true,
    );
    assert_numeric_case(
        required_numeric(&numerics, "av1-average-rate-fallback")?,
        [32, 10, 32, 10, 32, 10],
        [1, 1],
        [2, 10, 0, 0],
        [0, 0, 0],
        false,
    );
    assert_numeric_case(
        required_numeric(&numerics, "width-non-32-aligned")?,
        [34, 32, 64, 32, 34, 32],
        [25, 1],
        [2, 8, 0, 0],
        [0, 0, 0],
        true,
    );
    assert_numeric_case(
        required_numeric(&numerics, "height-only-nonalignment")?,
        [32, 18, 32, 18, 32, 18],
        [25, 1],
        [2, 8, 0, 0],
        [0, 0, 0],
        false,
    );
    assert_eq!(numerics.len(), 6);

    let workspace = workspace_root()?;
    assert_eq!(provenance.sources.len(), 6);
    for source in provenance.sources {
        let source_bytes = fs::read(workspace.join(&source.path))
            .with_context(|| format!("read pinned source {}", source.path))?;
        assert_eq!(sha256(&source_bytes), source.sha256, "{}", source.path);
    }
    Ok(())
}

fn case<'a>(manifest: &'a Manifest, id: &str) -> Result<&'a FixtureCase> {
    manifest
        .cases
        .iter()
        .find(|case| case.id == id)
        .ok_or_else(|| anyhow!("fixture case {id} is absent"))
}

fn assert_stream(
    actual: &StreamDisposition,
    ordinal: u8,
    codec: &str,
    decodable: bool,
    selected: bool,
    diagnostic: Option<&str>,
) {
    assert_eq!(actual.ordinal, ordinal);
    assert_eq!(actual.codec, codec);
    assert_eq!(actual.decodable, decodable);
    assert_eq!(actual.selected, selected);
    assert_eq!(actual.diagnostic.as_deref(), diagnostic);
}

fn required_numeric<'a>(
    cases: &'a BTreeMap<&str, &NumericCase>,
    id: &str,
) -> Result<&'a NumericCase> {
    cases
        .get(id)
        .copied()
        .ok_or_else(|| anyhow!("numeric fixture case {id} is absent"))
}

#[allow(clippy::too_many_arguments)]
fn assert_numeric_case(
    actual: &NumericCase,
    dimensions: [u16; 6],
    frame_rate: [u32; 2],
    frame: [i16; 4],
    audio: [u32; 3],
    enters_width_alignment_branch: bool,
) {
    assert_eq!(
        [
            actual.source_width,
            actual.source_height,
            actual.aligned_width,
            actual.aligned_height,
            actual.output_width,
            actual.output_height,
        ],
        dimensions
    );
    assert_eq!(
        [actual.frame_rate_numerator, actual.frame_rate_denominator],
        frame_rate
    );
    assert_eq!(i16::try_from(actual.frame_count), Ok(frame[0]));
    assert_eq!(i16::from(actual.bit_depth), frame[1]);
    assert_eq!(actual.alpha, frame[2] != 0);
    assert_eq!(i16::from(actual.rotation_quarter_turns), frame[3]);
    assert_eq!(u32::from(actual.audio_channels), audio[0]);
    assert_eq!(actual.audio_sample_rate, audio[1]);
    assert_eq!(actual.trimmed_audio_samples_per_channel, audio[2]);
    assert_eq!(
        actual.enters_width_alignment_branch,
        enters_width_alignment_branch
    );
    assert!(actual.decoded_video_sha256.is_none());
    assert!(actual.decoded_alpha_sha256.is_none());
    assert!(actual.decoded_audio_sha256.is_none());
}

fn assert_mp4_structural_seed(bytes: &[u8], id: &str) -> Result<()> {
    if bytes.get(4..8) != Some(b"ftyp".as_slice()) {
        bail!("MP4 fixture {id} is missing ftyp")
    }
    let marker = bytes
        .windows(b"zed-sim-general-demux-decode-structural-v1\0".len())
        .position(|window| window == b"zed-sim-general-demux-decode-structural-v1\0")
        .ok_or_else(|| anyhow!("MP4 fixture {id} is missing the structural marker"))?;
    let suffix = bytes
        .get(marker + b"zed-sim-general-demux-decode-structural-v1\0".len()..)
        .ok_or_else(|| anyhow!("MP4 fixture {id} marker exceeds bounds"))?;
    assert!(suffix.starts_with(id.as_bytes()));
    Ok(())
}

fn assert_webm_structural_seed(bytes: &[u8], id: &str) -> Result<()> {
    if !bytes.starts_with(b"\x1a\x45\xdf\xa3") {
        bail!("WebM fixture {id} is missing the EBML signature")
    }
    let marker = b"zed-sim-general-demux-decode-structural-v1\0";
    let offset = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| anyhow!("WebM fixture {id} is missing the structural marker"))?;
    let suffix = bytes
        .get(offset + marker.len()..)
        .ok_or_else(|| anyhow!("WebM fixture {id} marker exceeds bounds"))?;
    assert_eq!(suffix, id.as_bytes());
    Ok(())
}

fn coverage_bytes(files: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut coverage = Vec::new();
    for (path, bytes) in files {
        coverage
            .extend_from_slice(format!("{} {}  {path}\n", sha256(bytes), bytes.len()).as_bytes());
    }
    coverage
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
    visit(root, root, &mut output)?;
    Ok(output)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
