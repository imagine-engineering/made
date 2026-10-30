use serde_json::json;
use sha2::{Digest, Sha256};
use sim_model::{
    ArtifactAvailability, ArtifactIndex, ArtifactKey, ArtifactRecord, ArtifactRoot,
    GENERATED_LATENT_FORMATS, GENERATED_MODEL_FAMILY_REGISTRATIONS, LatentFormatIdentity,
    LatentFormatRegistry, ModelFamilyIdentity, ModelFamilyRegistry, ModelParsedFacts,
    ModelParsedFormatFact, ModelParsedTensorFact, ModelProbe, ModelStore, ParserLimits, PatchGraph,
    VAE_AUTOMATIC_ROW_ID, VAE_DIFFUSERS_ROW_ID, VAE_SELECTOR_BRANCH_COUNT,
    VAE_SELECTOR_CATALOG_ROWS, VAE_SELECTOR_ROW_COUNT, VAE_SELECTOR_SOURCE_PATH,
    VAE_SELECTOR_SOURCE_SHA256, VAE_UNBOUND_ROW_ID, VaeArchitectureError, VaeArchitectureRegistry,
    VaeBoundary, VaeCanonicalCompatibility, VaeCatalogRowKind, VaeDescriptor, VaeError,
    VaeExecutionTarget, VaeKernelProfile, VaeLoaderConfiguration, audio_vae_source_plan,
    structured_vae_source_plan, structured_vae_source_state_count, video_vae_source_plan,
};
use sim_tensor::{DType, DeviceId};
use sim_types::CancellationToken;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

const VAE_TILING_TASK: &str = "comfy-parity-vae-multidimensional-tiling";
const VAE_EXECUTION_TASK: &str = "comfy-parity-vae-execution-foundation";
const VAE_VIDEO_TASK: &str = "comfy-parity-vae-video-architectures";
const VAE_AUDIO_TASK: &str = "comfy-parity-vae-audio-architectures";
const VAE_STRUCTURED_TASK: &str = "comfy-parity-vae-structured-architectures";
const VAE_TILING_CASE_IDS: [&str; 7] = [
    "task353:source-digests-and-formulas",
    "task353:single-tile-direct-assignment",
    "task353:three-pass-feather-normalization",
    "task353:one-dimensional-reshape-and-channels",
    "task353:causal-three-dimensional-geometry",
    "task353:cancellation-oom-retry-atomicity",
    "task353:ownership-consolidation",
];

const VAE_TILING_CONTRACTS: [(&str, &str, &str, &str, &str); 12] = [
    (
        "conditioning-vae-tiling-sd-vae-encode-crop-pixels-d00009fc",
        "projects/comfy/ComfyUI/sim/sd.py",
        "vae_encode_crop_pixels",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "5a646c870fff8bb6fbed6843bf1b847606a50ad723de6d78f7d2f951d332f434",
    ),
    (
        "conditioning-vae-tiling-sd-decode-tiled-5f71cd3a",
        "projects/comfy/ComfyUI/sim/sd.py",
        "decode_tiled_",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "e03db6758f6451ad4ae8d6e07668f3e2bca571926c851f4aac8fec182d638cb1",
    ),
    (
        "conditioning-vae-tiling-sd-decode-tiled-1d-c55eeb16",
        "projects/comfy/ComfyUI/sim/sd.py",
        "decode_tiled_1d",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "35ab8b0f9f5a21810c72e6ded4fc46147f7ac087d704a996ed6384c45b217f53",
    ),
    (
        "conditioning-vae-tiling-sd-decode-tiled-3d-b3a56a11",
        "projects/comfy/ComfyUI/sim/sd.py",
        "decode_tiled_3d",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "53a82bb676bcf947460e2165a627533b9e85ee10fe07c5b8daa2081a5f639d26",
    ),
    (
        "conditioning-vae-tiling-sd-encode-tiled-c5f2e678",
        "projects/comfy/ComfyUI/sim/sd.py",
        "encode_tiled_",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "12ac8a9bda28824ba5e20c549b23841fc3574d21e5b40550f4853f9bc97435c7",
    ),
    (
        "conditioning-vae-tiling-sd-encode-tiled-1d-ac90cc2f",
        "projects/comfy/ComfyUI/sim/sd.py",
        "encode_tiled_1d",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "9975165d0ffc6a9b7bd468dd58dbdb9f00d74aa19ec79278bd0f4b634ba67c49",
    ),
    (
        "conditioning-vae-tiling-sd-encode-tiled-3d-c19b8274",
        "projects/comfy/ComfyUI/sim/sd.py",
        "encode_tiled_3d",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "34c73a6e76cc33eee6635dfb8fe7efb188934dce3d1683d35c92f237719ea3c1",
    ),
    (
        "conditioning-vae-tiling-sd-decode-tiled-c68762db",
        "projects/comfy/ComfyUI/sim/sd.py",
        "decode_tiled",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "adfd0ec4c73a37878efcb1713c66fd3afef9647a2a4fddb47c593f38c4bfdd1c",
    ),
    (
        "conditioning-vae-tiling-sd-encode-tiled-31fa96f6",
        "projects/comfy/ComfyUI/sim/sd.py",
        "encode_tiled",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "c3774774c919c193e91c88177db0da0e2ee7caa59cdf0fab9e587a70603ef886",
    ),
    (
        "conditioning-vae-tiling-utils-get-tiled-scale-steps-42e237ba",
        "projects/comfy/ComfyUI/sim/utils.py",
        "get_tiled_scale_steps",
        "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
        "baf721fd6df9b0367214e4f0e7783dddaba23c3a2627b597cf619e6190d08f33",
    ),
    (
        "conditioning-vae-tiling-utils-tiled-scale-multidim-7bf9ab2d",
        "projects/comfy/ComfyUI/sim/utils.py",
        "tiled_scale_multidim",
        "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
        "7c25f6cdfb945974faf66d076d297d6b271eaa13bf7274ccf5439f8c777415a5",
    ),
    (
        "conditioning-vae-tiling-utils-tiled-scale-87bb6712",
        "projects/comfy/ComfyUI/sim/utils.py",
        "tiled_scale",
        "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
        "bdff6f03f769db03a8bcec03f46b585c62676e0314043be3b1e26a50a1672e0e",
    ),
];

const VAE_IMAGE_TASK: &str = "comfy-parity-vae-image-architectures";
const VAE_IMAGE_CASE_IDS: [&str; 6] = [
    "task354:source-provenance-and-11-contracts",
    "task354:17-profile-manifests",
    "task354:encode-decode-equations",
    "task354:production-admission-dtypes-devices",
    "task354:cancellation-oom-retry-atomicity",
    "task354:ownership-consolidation",
];
const VAE_IMAGE_CONTRACTS: [(&str, &str, &str, &str, &str); 11] = [
    (
        "conditioning-vae-architecture-sd-autoencodingengine-5bca3e44",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.AutoencodingEngine@L509",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "ee292ffae46c5b07b350773eb077c86f5c45e5677306a56520087d2a6f51019e",
    ),
    (
        "conditioning-vae-architecture-sd-sim-taesd-taesd-taesd-d0ccd20f",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.sim.taesd.taesd.TAESD@L517",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "f372b326b0f9f23fe7e85ff586ab737e5dbfa1f8379d87e92a162df82cc70319",
    ),
    (
        "conditioning-vae-architecture-sd-stagea-1b767df0",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.StageA@L519",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "0bc6b6672b343f0d20c0f201c9e7fd9cc6e8d36f60f41b332c689f3fcc06264b",
    ),
    (
        "conditioning-vae-architecture-sd-stagec-coder-94678b59",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.StageC_coder@L528",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "6cc134881cfbd41ea2444db4b1c2bc5cf3e4ef6dfb731db2dafbe6b157617d0f",
    ),
    (
        "conditioning-vae-architecture-sd-stagec-coder-84281997",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.StageC_coder@L536",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "6cc134881cfbd41ea2444db4b1c2bc5cf3e4ef6dfb731db2dafbe6b157617d0f",
    ),
    (
        "conditioning-vae-architecture-sd-stagec-coder-4801dafb",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.StageC_coder@L543",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "6cc134881cfbd41ea2444db4b1c2bc5cf3e4ef6dfb731db2dafbe6b157617d0f",
    ),
    (
        "conditioning-vae-architecture-sd-autoencodingengine-b7150e80",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.AutoencodingEngine@L553",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "285934d7853d85038ff808e00931088e5884562caef2371a4713329ec9ce726a",
    ),
    (
        "conditioning-vae-architecture-sd-autoencoderkl-69aa0015",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.AutoencoderKL@L604",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "21406887c6bc9bdbf16927e9cefded6ee7922e74ab77168d3d03367db6a277b7",
    ),
    (
        "conditioning-vae-architecture-sd-autoencodingengine-9f13006f",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.AutoencodingEngine@L606",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "a3dc5c7ad32dd296e28d928ce899db4f721fed582c5ac3510b4c32a2a81df9fe",
    ),
    (
        "conditioning-vae-architecture-sd-sim-pixel-space-convert-pixelspaceconversionvae-3ec12255",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.sim.pixel_space_convert.PixelspaceConversionVAE@L800",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "93486c85cabde1276f74faf289fd8fae6d2faad4a6cb13a64cb2ffe095210f9d",
    ),
    (
        "conditioning-vae-architecture-sd-autoencoderkl-61f06f31",
        "projects/comfy/ComfyUI/sim/sd.py",
        "VAE.__init__.AutoencoderKL@L917",
        "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42",
        "bcbb41f758eafe25453967016a8532dc4bed7081518abd54372ffcd5bf763efc",
    ),
];

#[derive(Clone)]
struct SelectorFixture {
    row: &'static str,
    tensors: Vec<(&'static str, Vec<u64>, &'static str)>,
    metadata: BTreeMap<String, String>,
}

impl SelectorFixture {
    fn new(row: &'static str, tensors: Vec<(&'static str, Vec<u64>)>) -> Self {
        Self {
            row,
            tensors: tensors
                .into_iter()
                .map(|(name, shape)| (name, shape, "F32"))
                .collect(),
            metadata: BTreeMap::new(),
        }
    }

    fn with_dtype(mut self, tensor: &'static str, dtype: &'static str) -> Self {
        if let Some(entry) = self.tensors.iter_mut().find(|entry| entry.0 == tensor) {
            entry.2 = dtype;
        }
        self
    }

    fn probe(&self) -> Result<ModelProbe, Box<dyn Error>> {
        probe(&self.tensors, self.metadata.clone())
    }

    fn partial_probe(&self) -> Result<ModelProbe, Box<dyn Error>> {
        let mut tensors = self.tensors.clone();
        let target = match self.row {
            "conditioning-vae-selection-sd-l603-86be7303" => "post_quant_conv.weight",
            "conditioning-vae-selection-sd-l690-3abc5b4d" => "encoder.conv_out.conv.weight",
            "conditioning-vae-selection-sd-l731-01dbf62f" => {
                "decoder.upsamples.0.upsamples.0.residual.2.weight"
            }
            "conditioning-vae-selection-sd-l835-670a25d1"
            | "conditioning-vae-selection-sd-l840-410d45c2" => "decoder.1.weight",
            _ => tensors.first().ok_or("fixture has no tensor")?.0,
        };
        let entry = tensors
            .iter_mut()
            .find(|entry| entry.0 == target)
            .ok_or("partial fixture target is unavailable")?;
        match self.row {
            "conditioning-vae-selection-sd-l547-3d4531f9" => entry.1 = vec![1, 64, 0],
            "conditioning-vae-selection-sd-l559-295b67db" => entry.1 = vec![1, 32, 0, 3, 3],
            "conditioning-vae-selection-sd-l673-58c13833" => entry.1 = vec![1, 32, 0, 3, 3],
            "conditioning-vae-selection-sd-l835-670a25d1" => entry.1 = vec![8, 48, 0, 3, 3],
            "conditioning-vae-selection-sd-l840-410d45c2" => entry.1 = vec![8, 32, 0, 3, 3],
            _ => entry.1 = vec![0],
        }
        probe(&tensors, self.metadata.clone())
    }

    fn ambiguous_probe(&self) -> Result<ModelProbe, Box<dyn Error>> {
        let mut tensors = self.tensors.clone();
        let competitor = if tensors
            .iter()
            .any(|entry| entry.0 == "vquantizer.codebook.weight")
        {
            "taesd_decoder.1.weight"
        } else {
            "vquantizer.codebook.weight"
        };
        tensors.push((competitor, vec![1, 1], "F32"));
        probe(&tensors, self.metadata.clone())
    }
}

fn selector_fixtures() -> Vec<SelectorFixture> {
    vec![
        SelectorFixture::new(
            VAE_AUTOMATIC_ROW_ID,
            vec![("taesd_decoder.1.weight", vec![1, 4])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l504-48d322ed",
            vec![("decoder.mid.block_1.mix_factor", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l512-43630fab",
            vec![("taesd_decoder.1.weight", vec![1, 4])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l518-e78759a4",
            vec![("vquantizer.codebook.weight", vec![16, 4])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l527-8575102a",
            vec![("backbone.1.0.block.0.1.num_batches_tracked", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l535-1ad96396",
            vec![("blocks.11.num_batches_tracked", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l542-f7074db0",
            vec![("encoder.backbone.1.0.block.0.1.num_batches_tracked", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l546-cf72d403",
            vec![("decoder.conv_in.weight", vec![128, 4, 3, 3])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l547-3d4531f9",
            vec![("decoder.conv_in.weight", vec![128, 64, 3, 3])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l559-295b67db",
            vec![("decoder.conv_in.weight", vec![128, 32, 3, 3, 3])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l603-86be7303",
            vec![
                ("decoder.conv_in.weight", vec![128, 4, 3, 3]),
                ("post_quant_conv.weight", vec![4, 4, 1, 1]),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l609-82ce1c2f",
            vec![("decoder.layers.1.layers.0.beta", vec![64])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l636-d758fbe2",
            vec![("blocks.2.blocks.3.stack.5.weight", vec![1])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l651-b808b0d3",
            vec![(
                "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
                vec![512, 128, 3, 3, 3],
            )],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l673-58c13833",
            vec![("decoder.conv_in.conv.weight", vec![128, 32, 3, 3, 3])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l690-3abc5b4d",
            vec![
                ("decoder.conv_in.conv.weight", vec![128, 16, 3, 3, 3]),
                (
                    "decoder.mid_block.resnets.0.norm1.norm_layer.weight",
                    vec![128],
                ),
                ("encoder.conv_out.conv.weight", vec![32, 128, 3, 3, 3]),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l701-7056d652",
            vec![
                ("decoder.conv_in.conv.weight", vec![128, 4, 3, 3, 3]),
                ("post_quant_conv.weight", vec![4, 4, 1, 1, 1]),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l717-2e6a172a",
            vec![("decoder.unpatcher3d.wavelets", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l730-7d7bc483",
            vec![
                ("decoder.middle.0.residual.0.gamma", vec![160]),
                ("decoder.head.0.gamma", vec![160]),
                ("encoder.conv1.weight", vec![160, 3, 3, 3, 3]),
                ("decoder.head.2.weight", vec![3, 160, 3, 3, 3]),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l731-01dbf62f",
            vec![
                ("decoder.middle.0.residual.0.gamma", vec![160]),
                (
                    "decoder.upsamples.0.upsamples.0.residual.2.weight",
                    vec![160],
                ),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l762-d9946728",
            vec![("geo_decoder.cross_attn_decoder.ln_1.bias", vec![1])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l784-3c01e5be",
            vec![("vocoder.backbone.channel_layers.0.0.bias", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l799-186f31a5",
            vec![("pixel_space_vae", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l809-83aeacf2",
            vec![("vocoder.activation_post.downsample.lowpass.filter", vec![])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l828-58c549be",
            vec![
                ("decoder.22.bias", vec![8]),
                ("decoder.1.weight", vec![8, 16, 3, 3, 3]),
            ],
        )
        .with_dtype("decoder.1.weight", "F16"),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l835-670a25d1",
            vec![
                ("decoder.22.bias", vec![8]),
                ("decoder.1.weight", vec![8, 48, 3, 3, 3]),
            ],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l840-410d45c2",
            vec![
                ("decoder.22.bias", vec![12]),
                ("decoder.1.weight", vec![8, 32, 3, 3, 3]),
            ],
        ),
        {
            let mut fixture = SelectorFixture::new(
                "conditioning-vae-selection-sd-l856-a443d5ce",
                vec![("vocoder.resblocks.0.convs1.0.weight", vec![1])],
            );
            fixture.metadata.insert(
                "config".to_owned(),
                r#"{"audio_vae":{"model":{"params":{"ddconfig":{"z_channels":16},"sampling_rate":16000}},"preprocessing":{"stft":{"hop_length":160}}},"vocoder":{"upsample_rates":[5,4,2,2,2]}}"#
                    .to_owned(),
            );
            fixture
        },
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l874-13e203b0",
            vec![("decoder.layers.3.transformers.0.pre_norm.alpha", vec![1])],
        ),
        SelectorFixture::new(
            "conditioning-vae-selection-sd-l902-048f369c",
            vec![
                ("gs.base_offset_scale", vec![]),
                ("octree.out_proj.weight", vec![1]),
            ],
        ),
    ]
}

fn probe(
    tensors: &[(&str, Vec<u64>, &str)],
    metadata: BTreeMap<String, String>,
) -> Result<ModelProbe, Box<dyn Error>> {
    let tensors = tensors
        .iter()
        .map(|(name, shape, dtype)| {
            (
                (*name).to_owned(),
                ModelParsedTensorFact {
                    shape: shape.clone(),
                    storage_dtype: (*dtype).to_owned(),
                },
            )
        })
        .collect();
    Ok(ModelProbe::from_parsed_facts(ModelParsedFacts {
        tensors,
        formats: vec![ModelParsedFormatFact {
            identity: "safetensors".to_owned(),
            metadata,
        }],
    })?)
}

fn workspace() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is unavailable")?
        .to_path_buf())
}

#[test]
fn val_vae_001_selector_rows() -> Result<(), Box<dyn Error>> {
    let registry = VaeArchitectureRegistry::checked()?;
    let cancellation = CancellationToken::default();
    let fixtures = selector_fixtures();
    assert_eq!(fixtures.len(), VAE_SELECTOR_BRANCH_COUNT);
    assert_eq!(VAE_SELECTOR_CATALOG_ROWS.len(), VAE_SELECTOR_ROW_COUNT);

    let expected_rows = VAE_SELECTOR_CATALOG_ROWS
        .iter()
        .filter(|row| matches!(row.kind, VaeCatalogRowKind::SelectionBranch))
        .map(|row| row.contract_id)
        .collect::<BTreeSet<_>>();
    let fixture_rows = fixtures
        .iter()
        .map(|fixture| fixture.row)
        .collect::<BTreeSet<_>>();
    assert_eq!(fixture_rows, expected_rows);

    let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;

    let mut validated = Vec::new();
    for fixture in &fixtures {
        let selection = registry.select(&fixture.probe()?, &cancellation)?;
        assert_eq!(
            selection.canonical_compatibility(),
            selection.profile().canonical_compatibility()
        );
        assert_eq!(
            selection.supported_dtypes(),
            selection.profile().supported_dtypes()
        );
        assert_eq!(
            selection.latent_dimensions(),
            selection.profile().latent_dimensions()
        );
        assert_eq!(
            selection.boundary(),
            match selection.profile().expected_boundary_kind() {
                sim_model::VaeBoundaryKind::Image => sim_model::VaeBoundaryDomain::Image,
                sim_model::VaeBoundaryKind::Video => sim_model::VaeBoundaryDomain::Video,
                sim_model::VaeBoundaryKind::Audio => sim_model::VaeBoundaryDomain::Audio,
                sim_model::VaeBoundaryKind::StructuredOutput => {
                    sim_model::VaeBoundaryDomain::Structured
                }
            }
        );
        assert_eq!(
            selection.target_latent_channels(),
            selection
                .profile()
                .target_latent_channels()
                .or(selection.latent_channels())
        );
        assert!(selection.trace().contains(&fixture.row));
        let registered_adapter_profile = matches!(
            selection.profile(),
            sim_model::VaeKernelProfile::TemporalAutoencodingEngineV1
                | sim_model::VaeKernelProfile::TaesdV1
                | sim_model::VaeKernelProfile::StableCascadeStageAV1
                | sim_model::VaeKernelProfile::StableCascadeStageCEncoderV1
                | sim_model::VaeKernelProfile::StableCascadeStageCPreviewerV1
                | sim_model::VaeKernelProfile::StableCascadeStageCCombinedV1
                | sim_model::VaeKernelProfile::HunyuanImageV1
                | sim_model::VaeKernelProfile::AutoencoderKlV1
                | sim_model::VaeKernelProfile::AutoencoderKlX4V1
                | sim_model::VaeKernelProfile::AutoencoderKlBatchNormV1
                | sim_model::VaeKernelProfile::ExplicitAutoencoderKlV1
                | sim_model::VaeKernelProfile::AutoencodingEngineV1
                | sim_model::VaeKernelProfile::AutoencodingEngineX4V1
                | sim_model::VaeKernelProfile::AutoencodingEngineBatchNormV1
                | sim_model::VaeKernelProfile::PixelSpaceV1
                | sim_model::VaeKernelProfile::TaeHvWan22V1
                | sim_model::VaeKernelProfile::TaeHvLtx2V1
                | sim_model::VaeKernelProfile::LightTaeHv15V1
                | sim_model::VaeKernelProfile::TaeHvHunyuanV1
                | sim_model::VaeKernelProfile::LightTaeWan21V1
                | sim_model::VaeKernelProfile::HunyuanImageRefinerV1
                | sim_model::VaeKernelProfile::HunyuanVideoRefinerV1
                | sim_model::VaeKernelProfile::Causal3dV1
                | sim_model::VaeKernelProfile::CogVideoXV1
                | sim_model::VaeKernelProfile::CosmosV1
                | sim_model::VaeKernelProfile::MochiV1
                | sim_model::VaeKernelProfile::LtxVideoV0 { .. }
                | sim_model::VaeKernelProfile::LtxVideoV1 { .. }
                | sim_model::VaeKernelProfile::LtxVideoV2 { .. }
                | sim_model::VaeKernelProfile::Wan21V1
                | sim_model::VaeKernelProfile::Wan22V1
                | sim_model::VaeKernelProfile::AudioOobleck44KhzV1
                | sim_model::VaeKernelProfile::AudioOobleck48KhzV1
                | sim_model::VaeKernelProfile::MusicDcaeV1
                | sim_model::VaeKernelProfile::MmAudio16KhzV1
                | sim_model::VaeKernelProfile::LtxAudioV1
                | sim_model::VaeKernelProfile::StableAudio3DeepV1
                | sim_model::VaeKernelProfile::StableAudio3ShallowV1
                | sim_model::VaeKernelProfile::HunyuanShapeV1
                | sim_model::VaeKernelProfile::TripoSplatV1
        );
        if registered_adapter_profile {
            selection.ensure_native_builder_available()?;
        } else {
            assert!(matches!(
                selection.ensure_native_builder_available(),
                Err(VaeArchitectureError::ArchitectureUnavailable { .. })
            ));
        }
        match registry.select(&fixture.partial_probe()?, &cancellation) {
            Err(VaeArchitectureError::Partial { .. }) if fixture.row == VAE_AUTOMATIC_ROW_ID => {}
            Err(VaeArchitectureError::Partial { row, .. }) => assert_eq!(row, fixture.row),
            Err(VaeArchitectureError::NoMatch { .. }) if fixture.row == VAE_AUTOMATIC_ROW_ID => {}
            result => panic!("{} partial fixture returned {result:?}", fixture.row),
        }
        match registry.select(&fixture.ambiguous_probe()?, &cancellation) {
            Err(VaeArchitectureError::Ambiguous { rows }) => {
                if fixture.row != VAE_AUTOMATIC_ROW_ID {
                    assert!(
                        rows.contains(&fixture.row),
                        "{rows:?} omits {}",
                        fixture.row
                    );
                }
            }
            result => panic!("{} ambiguous fixture returned {result:?}", fixture.row),
        }
        let target_outcome = match registry.intended_target(
            &selection,
            &family_registry,
            &latent_registry,
            &cancellation,
        ) {
            Ok(target) => {
                registry.validate_target(
                    &selection,
                    &target,
                    &family_registry,
                    &latent_registry,
                    &cancellation,
                )?;
                "canonical_target_passed"
            }
            Err(VaeArchitectureError::CanonicalTargetUnavailable { .. }) => {
                assert!(matches!(
                    selection.canonical_compatibility(),
                    VaeCanonicalCompatibility::Unavailable(_)
                ));
                "canonical_latent_unavailable"
            }
            Err(VaeArchitectureError::CanonicalFamilyUnavailable { .. }) => {
                assert!(matches!(
                    selection.canonical_compatibility(),
                    VaeCanonicalCompatibility::Exact(_)
                ));
                "canonical_family_unavailable"
            }
            Err(error) => return Err(error.into()),
        };
        let mut wrong_family_rejected = false;
        for family in family_registry.definitions_in_source_order() {
            if !family.supported_dtypes.contains(&DType::F32)
                || !family
                    .supported_devices
                    .contains(&sim_types::DeviceKind::Cpu)
            {
                continue;
            }
            let candidate = VaeExecutionTarget::new(
                ModelFamilyIdentity::new(
                    family.feature_id,
                    family.identifier,
                    family.architecture_version,
                )?,
                LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
                DType::F32,
                DeviceId::CPU,
            );
            if matches!(
                registry.validate_target(
                    &selection,
                    &candidate,
                    &family_registry,
                    &latent_registry,
                    &cancellation,
                ),
                Err(VaeArchitectureError::ProfileLatentMismatch { .. })
                    | Err(VaeArchitectureError::CanonicalTargetUnavailable { .. })
            ) {
                wrong_family_rejected = true;
                break;
            }
        }
        assert!(
            wrong_family_rejected,
            "{} has no canonical wrong-family rejection fixture",
            fixture.row
        );
        validated.push(json!({
            "contract_id": fixture.row,
            "architecture": selection.architecture().as_str(),
            "profile": format!("{:?}", selection.profile()),
            "positive": "passed",
            "partial": "passed",
            "ambiguous": "passed",
            "wrong_family": "passed",
            "intended_target": target_outcome,
            "builder": if registered_adapter_profile {
                "registered_native_image_adapter_constructor"
            } else {
                "typed_unavailable"
            },
        }));
    }

    let mut first = SelectorFixture::new(
        "conditioning-vae-selection-sd-l651-b808b0d3",
        vec![(
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
            vec![512, 128, 3, 3, 3],
        )],
    );
    first.metadata.insert(
        "config".to_owned(),
        r#"{"vae":{"b":{"x":2,"a":1},"a":0}}"#.to_owned(),
    );
    let mut second = first.clone();
    second.metadata.insert(
        "config".to_owned(),
        r#"{"vae":{"a":0,"b":{"a":1,"x":2}}}"#.to_owned(),
    );
    let first = registry.select(&first.probe()?, &cancellation)?;
    let second = registry.select(&second.probe()?, &cancellation)?;
    assert_eq!(first.loader_configuration(), second.loader_configuration());
    let mut changed = SelectorFixture::new(
        "conditioning-vae-selection-sd-l651-b808b0d3",
        vec![(
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
            vec![512, 128, 3, 3, 3],
        )],
    );
    changed.metadata.insert(
        "config".to_owned(),
        r#"{"vae":{"a":1,"b":{"a":1,"x":2}}}"#.to_owned(),
    );
    assert_ne!(
        first.loader_configuration(),
        registry
            .select(&changed.probe()?, &cancellation)?
            .loader_configuration()
    );
    let mut oversized = SelectorFixture::new(
        "conditioning-vae-selection-sd-l651-b808b0d3",
        vec![(
            "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
            vec![512, 128, 3, 3, 3],
        )],
    );
    oversized.metadata.insert(
        "config".to_owned(),
        format!(r#"{{"vae":{{"padding":"{}"}}}}"#, "x".repeat(256 * 1024)),
    );
    assert!(matches!(
        registry.select(&oversized.probe()?, &cancellation),
        Err(VaeArchitectureError::ConfigurationLimit { .. })
    ));

    assert!(matches!(
        registry.select(
            &probe(&[], BTreeMap::new())?,
            &CancellationToken::default()
        ),
        Err(VaeArchitectureError::NoMatch { unbound_row }) if unbound_row == VAE_UNBOUND_ROW_ID
    ));
    let converted = registry.select(
        &probe(
            &[
                (
                    "decoder.up_blocks.0.resnets.0.norm1.weight",
                    vec![128],
                    "F32",
                ),
                ("decoder.conv_in.weight", vec![128, 4, 3, 3], "F32"),
            ],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert!(converted.trace().contains(&VAE_DIFFUSERS_ROW_ID));

    let workspace = workspace()?;
    let source = fs::read(workspace.join(VAE_SELECTOR_SOURCE_PATH))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&source)),
        VAE_SELECTOR_SOURCE_SHA256
    );
    let catalog = fs::read_to_string(
        workspace.join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let catalog_rows = parse_catalog(&catalog)?;
    const LOADER_TASK: &str = "comfy-parity-vae-domain-loader-foundation";
    let vae_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == LOADER_TASK))
        .collect::<Vec<_>>();
    assert_eq!(vae_rows.len(), VAE_SELECTOR_ROW_COUNT);
    let mut contract_evidence = Vec::with_capacity(VAE_SELECTOR_ROW_COUNT);
    for row in VAE_SELECTOR_CATALOG_ROWS {
        let fields = vae_rows
            .iter()
            .find(|fields| fields.first().is_some_and(|id| id == row.contract_id))
            .ok_or_else(|| format!("catalog row {} is missing", row.contract_id))?;
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], row.kind.catalog_name());
        assert_eq!(fields[2], VAE_SELECTOR_SOURCE_PATH);
        assert_eq!(fields[3], row.source_symbol);
        assert_eq!(fields[4], row.source_ordinal.to_string());
        assert_eq!(fields[5], VAE_SELECTOR_SOURCE_SHA256);
        assert_eq!(fields[6], row.symbol_sha256);
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], LOADER_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(
            fields[10],
            if matches!(row.kind, VaeCatalogRowKind::ArchitectureUnavailable) {
                "native_fail_closed"
            } else {
                "native_rust"
            }
        );
        assert_eq!(fields[14], "VAL-VAE-001");
        let case_ids = if row.contract_id == VAE_DIFFUSERS_ROW_ID {
            vec![
                format!("{}:positive", row.contract_id),
                format!("{}:malformed", row.contract_id),
                format!("{}:cancellation", row.contract_id),
            ]
        } else if row.contract_id == VAE_UNBOUND_ROW_ID {
            vec![format!("{}:typed-no-match", row.contract_id)]
        } else {
            vec![
                format!("{}:positive", row.contract_id),
                format!("{}:partial", row.contract_id),
                format!("{}:ambiguous", row.contract_id),
                format!("{}:wrong-family", row.contract_id),
            ]
        };
        contract_evidence.push(json!({
            "contract_id": row.contract_id,
            "task_id": LOADER_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": case_ids,
        }));
    }
    let loader_contract_count = contract_evidence.len();
    let execution_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == VAE_EXECUTION_TASK))
        .collect::<Vec<_>>();
    assert_eq!(execution_rows.len(), 1);
    let execution_fields = execution_rows
        .first()
        .copied()
        .ok_or("VAE execution contract is unavailable")?;
    assert_eq!(execution_fields.len(), 15);
    assert_eq!(
        execution_fields[0],
        "conditioning-model-execution-sd-vae-6e4631bd"
    );
    assert_eq!(execution_fields[1], "model_execution");
    assert_eq!(execution_fields[2], "projects/comfy/ComfyUI/sim/sd.py");
    assert_eq!(execution_fields[3], "VAE");
    assert_eq!(execution_fields[5], VAE_SELECTOR_SOURCE_SHA256);
    assert_eq!(
        execution_fields[6],
        "b84589afa030b5865c809774f949922d464a94a26375957f2da423b4a27524d4"
    );
    assert_eq!(execution_fields[7], "sim_model::vae");
    assert_eq!(execution_fields[8], VAE_EXECUTION_TASK);
    assert_eq!(execution_fields[9], "sim_model::vae::tests");
    assert_eq!(execution_fields[10], "native_rust");
    assert_eq!(execution_fields[14], "VAL-VAE-001");

    let execution_family = family_registry
        .definitions_in_source_order()
        .into_iter()
        .find(|definition| definition.identifier == "CogVideoX_T2V")
        .ok_or("canonical CogVideoX family is unavailable")?;
    let execution_target = VaeExecutionTarget::new(
        ModelFamilyIdentity::new(
            execution_family.feature_id,
            execution_family.identifier,
            execution_family.architecture_version,
        )?,
        LatentFormatIdentity::new(
            execution_family.latent_feature_id,
            execution_family.latent_identifier,
        )?,
        DType::F32,
        DeviceId::CPU,
    );
    let execution_fixture = selector_fixtures()
        .into_iter()
        .find(|fixture| fixture.row == "conditioning-vae-selection-sd-l690-3abc5b4d")
        .ok_or("CogVideoX VAE fixture is unavailable")?;
    let execution_selection = registry.select_for_target(
        &execution_fixture.probe()?,
        &execution_target,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    let execution_artifact = ArtifactRecord {
        key: ArtifactKey::new("models", "cog-vae.safetensors")?,
        namespace: "vae".to_owned(),
        canonical_path: PathBuf::from("/verified/models/cog-vae.safetensors"),
        byte_size: 1,
        modified_nanoseconds: 1,
        sha256: "a".repeat(64),
        availability: ArtifactAvailability::Present,
    };
    let execution_patch =
        PatchGraph::checked_semantic(execution_artifact.sha256.clone(), Vec::new())?.identity();
    let execution_descriptor = VaeDescriptor::checked_selection(
        &execution_artifact,
        &execution_selection,
        &execution_target,
        &family_registry,
        &latent_registry,
        execution_patch.clone(),
        VaeBoundary::video(3)?,
        [0.0, 1.0],
        &cancellation,
    )?;
    assert_eq!(
        execution_descriptor.identity().family(),
        execution_target.family()
    );
    assert_eq!(
        execution_descriptor.identity().latent_format(),
        execution_target.latent_format()
    );
    let encoded_execution_identity = serde_json::to_value(execution_descriptor.identity())?;
    let decoded_execution_identity =
        serde_json::from_value::<sim_model::VaeIdentity>(encoded_execution_identity)?;
    assert_eq!(decoded_execution_identity, *execution_descriptor.identity());
    assert!(matches!(
        VaeDescriptor::checked_selection(
            &execution_artifact,
            &execution_selection,
            &execution_target,
            &family_registry,
            &latent_registry,
            execution_patch,
            VaeBoundary::image(3)?,
            [0.0, 1.0],
            &cancellation,
        ),
        Err(VaeError::SelectionBoundaryMismatch { .. })
    ));
    contract_evidence.push(json!({
        "contract_id": execution_fields[0],
        "task_id": VAE_EXECUTION_TASK,
        "source_sha256": execution_fields[5],
        "symbol_sha256": execution_fields[6],
        "status": "passed",
        "case_ids": [
            "vae-execution:canonical-selection-binding",
            "vae-execution:identity-round-trip",
            "vae-execution:typed-boundary-rejection",
        ],
    }));
    let tiling_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == VAE_TILING_TASK))
        .collect::<Vec<_>>();
    assert_eq!(tiling_rows.len(), VAE_TILING_CONTRACTS.len());
    for (contract_id, source_path, source_symbol, source_sha256, symbol_sha256) in
        VAE_TILING_CONTRACTS
    {
        let fields = tiling_rows
            .iter()
            .find(|fields| fields.first().is_some_and(|id| id == contract_id))
            .ok_or_else(|| format!("catalog row {contract_id} is missing"))?;
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], "vae_tiling");
        assert_eq!(fields[2], source_path);
        assert_eq!(fields[3], source_symbol);
        assert_eq!(fields[5], source_sha256);
        assert_eq!(fields[6], symbol_sha256);
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], VAE_TILING_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-VAE-001");
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join(source_path))?)
            ),
            source_sha256
        );
        contract_evidence.push(json!({
            "contract_id": contract_id,
            "task_id": VAE_TILING_TASK,
            "source_sha256": source_sha256,
            "symbol_sha256": symbol_sha256,
            "status": "passed",
            "case_ids": [
                format!("{contract_id}:source-exact"),
                format!("{contract_id}:typed-invalid"),
            ],
        }));
    }

    let image_fixture_path =
        "crates/sim_test_support/fixtures/models/vae-image/architecture-checkpoints.json";
    let image_provenance_path = "crates/sim_test_support/fixtures/models/vae-image/provenance.json";
    let image_fixture_bytes = fs::read(workspace.join(image_fixture_path))?;
    let image_provenance_bytes = fs::read(workspace.join(image_provenance_path))?;
    let image_fixture: serde_json::Value = serde_json::from_slice(&image_fixture_bytes)?;
    assert_eq!(image_fixture["schema_version"], 1);
    assert_eq!(
        image_fixture["fixture_id"],
        "sim-native-image-vae-source-checkpoints-v1"
    );
    assert_eq!(
        image_fixture["provenance_sha256"],
        format!("{:x}", Sha256::digest(&image_provenance_bytes))
    );
    let image_cases = image_fixture["cases"]
        .as_array()
        .ok_or("image VAE fixture cases are unavailable")?;
    assert_eq!(image_cases.len(), 17);
    let image_case_ids = image_cases
        .iter()
        .map(|case| {
            case["id"]
                .as_str()
                .map(str::to_owned)
                .ok_or("image VAE fixture case id is unavailable")
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    assert_eq!(image_case_ids.len(), 17);
    let mut image_equation_checkpoint_count = 0;
    let mut unique_image_equation_checkpoints = BTreeSet::new();
    let image_case_evidence = image_cases
        .iter()
        .map(|case| {
            let id = case["id"]
                .as_str()
                .ok_or("image VAE fixture case id is unavailable")?;
            let profile = case["profile"]
                .as_str()
                .ok_or("image VAE fixture profile is unavailable")?;
            let encode = case["encode"]
                .as_str()
                .ok_or("image VAE fixture encode status is unavailable")?;
            let decode = case["decode"]
                .as_str()
                .ok_or("image VAE fixture decode status is unavailable")?;
            assert!(matches!(encode, "available" | "typed_unavailable"));
            assert!(matches!(decode, "available" | "typed_unavailable"));
            let state_checkpoints = case["state_checkpoints"]
                .as_array()
                .ok_or("image VAE fixture state checkpoints are unavailable")?;
            assert!(
                !state_checkpoints.is_empty(),
                "{id} has no state checkpoint"
            );
            let equations = case["equation_checkpoints"]
                .as_array()
                .ok_or("image VAE fixture equation checkpoints are unavailable")?
                .iter()
                .map(|equation| {
                    equation
                        .as_str()
                        .map(str::to_owned)
                        .ok_or("image VAE equation checkpoint is unavailable")
                })
                .collect::<Result<Vec<_>, _>>()?;
            assert!(!equations.is_empty(), "{id} has no equation checkpoint");
            assert_eq!(
                equations.iter().collect::<BTreeSet<_>>().len(),
                equations.len(),
                "{id} repeats an equation checkpoint"
            );
            image_equation_checkpoint_count += equations.len();
            unique_image_equation_checkpoints.extend(equations.iter().cloned());
            Ok(json!({
                "id": id,
                "profile": profile,
                "encode": encode,
                "decode": decode,
                "state_checkpoint_count": state_checkpoints.len(),
                "equation_checkpoints": equations,
            }))
        })
        .collect::<Result<Vec<_>, &str>>()?;
    assert_eq!(image_equation_checkpoint_count, 45);
    assert_eq!(unique_image_equation_checkpoints.len(), 39);

    let image_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == VAE_IMAGE_TASK))
        .collect::<Vec<_>>();
    assert_eq!(image_rows.len(), VAE_IMAGE_CONTRACTS.len());
    for (contract_id, source_path, source_symbol, source_sha256, symbol_sha256) in
        VAE_IMAGE_CONTRACTS
    {
        let fields = image_rows
            .iter()
            .find(|fields| fields.first().is_some_and(|id| id == contract_id))
            .ok_or_else(|| format!("catalog row {contract_id} is missing"))?;
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], "vae_architecture");
        assert_eq!(fields[2], source_path);
        assert_eq!(fields[3], source_symbol);
        assert_eq!(fields[5], source_sha256);
        assert_eq!(fields[6], symbol_sha256);
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], VAE_IMAGE_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-VAE-001");
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join(source_path))?)
            ),
            source_sha256
        );
        let case_ids = image_cases
            .iter()
            .filter(|case| {
                case["catalog_contract_ids"]
                    .as_array()
                    .is_some_and(|contracts| contracts.iter().any(|value| value == contract_id))
            })
            .map(|case| {
                case["id"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("image VAE fixture case id is unavailable")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert!(
            !case_ids.is_empty(),
            "{contract_id} has no image fixture case"
        );
        contract_evidence.push(json!({
            "contract_id": contract_id,
            "task_id": VAE_IMAGE_TASK,
            "source_sha256": source_sha256,
            "symbol_sha256": symbol_sha256,
            "status": "passed",
            "case_ids": case_ids,
        }));
    }

    let video_fixture_path =
        "crates/sim_test_support/fixtures/models/vae-video/architecture-checkpoints.json";
    let video_provenance_path = "crates/sim_test_support/fixtures/models/vae-video/provenance.json";
    let video_fixture_bytes = fs::read(workspace.join(video_fixture_path))?;
    let video_provenance_bytes = fs::read(workspace.join(video_provenance_path))?;
    let video_fixture: serde_json::Value = serde_json::from_slice(&video_fixture_bytes)?;
    assert_eq!(video_fixture["schema_version"], 1);
    assert_eq!(
        video_fixture["fixture_id"],
        "sim-native-video-vae-source-checkpoints-v1"
    );
    assert_eq!(
        video_fixture["provenance_sha256"],
        format!("{:x}", Sha256::digest(&video_provenance_bytes))
    );
    let video_cases = video_fixture["cases"]
        .as_array()
        .ok_or("video VAE fixture cases are unavailable")?;
    assert_eq!(video_cases.len(), 17);
    let mut video_case_ids = BTreeSet::new();
    let mut video_contract_case_ids = BTreeMap::<String, Vec<String>>::new();
    let mut video_equation_checkpoint_count = 0_usize;
    let mut unique_video_equation_checkpoints = BTreeSet::new();
    let mut video_case_evidence = Vec::new();
    for case in video_cases {
        let id = case["id"]
            .as_str()
            .ok_or("video VAE fixture case id is unavailable")?;
        assert!(video_case_ids.insert(id.to_owned()));
        let profile_name = case["profile"]
            .as_str()
            .ok_or("video VAE fixture profile is unavailable")?;
        let profile = match profile_name {
            "HunyuanImageRefinerV1" => VaeKernelProfile::HunyuanImageRefinerV1,
            "MochiV1" => VaeKernelProfile::MochiV1,
            "LtxVideoV0" => VaeKernelProfile::LtxVideoV0 {
                configuration_sha256: None,
            },
            "LtxVideoV1" => VaeKernelProfile::LtxVideoV1 {
                configuration_sha256: None,
            },
            "LtxVideoV2" => VaeKernelProfile::LtxVideoV2 {
                configuration_sha256: None,
            },
            "HunyuanVideoRefinerV1" => VaeKernelProfile::HunyuanVideoRefinerV1,
            "CogVideoXV1" => VaeKernelProfile::CogVideoXV1,
            "Causal3dV1" => VaeKernelProfile::Causal3dV1,
            "CosmosV1" => VaeKernelProfile::CosmosV1,
            "Wan21V1" => VaeKernelProfile::Wan21V1,
            "Wan22V1" => VaeKernelProfile::Wan22V1,
            "TaeHvWan22V1" => VaeKernelProfile::TaeHvWan22V1,
            "TaeHvLtx2V1" => VaeKernelProfile::TaeHvLtx2V1,
            "LightTaeHv15V1" => VaeKernelProfile::LightTaeHv15V1,
            "TaeHvHunyuanV1" => VaeKernelProfile::TaeHvHunyuanV1,
            "LightTaeWan21V1" => VaeKernelProfile::LightTaeWan21V1,
            other => return Err(format!("unknown video VAE fixture profile {other}").into()),
        };
        let plan = video_vae_source_plan(&profile)?;
        assert_eq!(case["temporal_ratio"].as_u64(), Some(plan.temporal_ratio()));
        assert_eq!(case["spatial_ratio"].as_u64(), Some(plan.spatial_ratio()));
        let state_checkpoints = case["state_checkpoints"]
            .as_array()
            .ok_or("video VAE fixture state checkpoints are unavailable")?;
        assert!(
            !state_checkpoints.is_empty(),
            "{id} has no state checkpoint"
        );
        for checkpoint in state_checkpoints {
            let name = checkpoint["name"]
                .as_str()
                .ok_or("video VAE checkpoint name is unavailable")?;
            let rank = checkpoint["rank"]
                .as_u64()
                .ok_or("video VAE checkpoint rank is unavailable")?;
            assert!(
                plan.state_checkpoints()
                    .iter()
                    .any(|state| state.name == name && u64::from(state.rank) == rank),
                "{id} checkpoint {name} rank {rank} is not implemented"
            );
        }
        let equations = case["equation_checkpoints"]
            .as_array()
            .ok_or("video VAE equation checkpoints are unavailable")?
            .iter()
            .map(|equation| {
                equation
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("video VAE equation checkpoint is unavailable")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            equations,
            plan.equation_checkpoints(),
            "{id} equation checkpoints diverged"
        );
        video_equation_checkpoint_count += equations.len();
        unique_video_equation_checkpoints.extend(equations.iter().cloned());
        let contracts = case["catalog_contract_ids"]
            .as_array()
            .ok_or("video VAE catalog contracts are unavailable")?;
        assert!(!contracts.is_empty(), "{id} has no catalog contract");
        for contract in contracts {
            let contract = contract
                .as_str()
                .ok_or("video VAE catalog contract is unavailable")?;
            video_contract_case_ids
                .entry(contract.to_owned())
                .or_default()
                .push(id.to_owned());
        }
        video_case_evidence.push(json!({
            "id": id,
            "profile": profile_name,
            "state_checkpoint_count": state_checkpoints.len(),
            "equation_checkpoints": equations,
            "native_encode_path": "implemented",
            "native_decode_path": "implemented",
            "validation_scope": "immutable source topology and named native semantic equation tests",
        }));
    }
    assert_eq!(video_case_ids.len(), 17);
    assert_eq!(video_contract_case_ids.len(), 12);
    let video_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == VAE_VIDEO_TASK))
        .collect::<Vec<_>>();
    assert_eq!(video_rows.len(), 12);
    for fields in video_rows {
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], "vae_architecture");
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], VAE_VIDEO_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-VAE-001");
        let contract_id = fields
            .first()
            .ok_or("video VAE contract id is unavailable")?;
        let case_ids = video_contract_case_ids
            .get(contract_id)
            .ok_or_else(|| format!("video VAE contract {contract_id} has no fixture case"))?;
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join(&fields[2]))?)
            ),
            fields[5]
        );
        contract_evidence.push(json!({
            "contract_id": contract_id,
            "task_id": VAE_VIDEO_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": case_ids,
        }));
    }

    let audio_fixture_path =
        "crates/sim_test_support/fixtures/models/vae-audio/architecture-checkpoints.json";
    let audio_provenance_path = "crates/sim_test_support/fixtures/models/vae-audio/provenance.json";
    let audio_fixture_bytes = fs::read(workspace.join(audio_fixture_path))?;
    let audio_provenance_bytes = fs::read(workspace.join(audio_provenance_path))?;
    let audio_fixture: serde_json::Value = serde_json::from_slice(&audio_fixture_bytes)?;
    assert_eq!(audio_fixture["schema_version"], 1);
    assert_eq!(
        audio_fixture["fixture_id"],
        "sim-native-audio-vae-source-checkpoints-v1"
    );
    assert_eq!(
        audio_fixture["provenance_sha256"],
        format!("{:x}", Sha256::digest(&audio_provenance_bytes))
    );
    let audio_cases = audio_fixture["cases"]
        .as_array()
        .ok_or("audio VAE fixture cases are unavailable")?;
    assert_eq!(audio_cases.len(), 7);
    let mut audio_case_ids = BTreeSet::new();
    let mut audio_contract_case_ids = BTreeMap::<String, Vec<String>>::new();
    let mut audio_equation_checkpoint_count = 0_usize;
    let mut unique_audio_equation_checkpoints = BTreeSet::new();
    let mut audio_case_evidence = Vec::new();
    for case in audio_cases {
        let id = case["id"]
            .as_str()
            .ok_or("audio VAE fixture case id is unavailable")?;
        assert!(audio_case_ids.insert(id.to_owned()));
        let profile_name = case["profile"]
            .as_str()
            .ok_or("audio VAE fixture profile is unavailable")?;
        let profile = match profile_name {
            "AudioOobleck44KhzV1" => VaeKernelProfile::AudioOobleck44KhzV1,
            "AudioOobleck48KhzV1" => VaeKernelProfile::AudioOobleck48KhzV1,
            "MusicDcaeV1" => VaeKernelProfile::MusicDcaeV1,
            "MmAudio16KhzV1" => VaeKernelProfile::MmAudio16KhzV1,
            "LtxAudioV1" => VaeKernelProfile::LtxAudioV1,
            "StableAudio3DeepV1" => VaeKernelProfile::StableAudio3DeepV1,
            "StableAudio3ShallowV1" => VaeKernelProfile::StableAudio3ShallowV1,
            other => return Err(format!("unknown audio VAE fixture profile {other}").into()),
        };
        let plan = audio_vae_source_plan(&profile)?;
        assert_eq!(
            case["input_sample_rate"].as_u64(),
            Some(u64::from(plan.input_sample_rate()))
        );
        assert_eq!(
            case["output_sample_rate"].as_u64(),
            Some(u64::from(plan.output_sample_rate()))
        );
        assert_eq!(
            (
                case["sample_ratio_numerator"].as_u64(),
                case["sample_ratio_denominator"].as_u64()
            ),
            (Some(plan.sample_ratio().0), Some(plan.sample_ratio().1))
        );
        assert_eq!(
            case["latent_dimensions"].as_u64(),
            Some(u64::from(plan.latent_dimensions()))
        );
        let equations = case["equation_checkpoints"]
            .as_array()
            .ok_or("audio VAE equation checkpoints are unavailable")?
            .iter()
            .map(|equation| {
                equation
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("audio VAE equation checkpoint is unavailable")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(equations, plan.equation_checkpoints());
        audio_equation_checkpoint_count += equations.len();
        unique_audio_equation_checkpoints.extend(equations.iter().cloned());
        let contracts = case["catalog_contract_ids"]
            .as_array()
            .ok_or("audio VAE catalog contracts are unavailable")?;
        assert!(!contracts.is_empty(), "{id} has no catalog contract");
        for contract in contracts {
            let contract = contract
                .as_str()
                .ok_or("audio VAE catalog contract is unavailable")?;
            audio_contract_case_ids
                .entry(contract.to_owned())
                .or_default()
                .push(id.to_owned());
        }
        audio_case_evidence.push(json!({
            "id": id,
            "profile": profile_name,
            "equation_checkpoints": equations,
            "native_encode_path": "implemented",
            "native_decode_path": "implemented",
            "validation_scope": "immutable source topology and named native semantic equation tests",
        }));
    }
    assert_eq!(audio_case_ids.len(), 7);
    assert_eq!(audio_contract_case_ids.len(), 5);
    let audio_rows = catalog_rows
        .iter()
        .filter(|fields| fields.get(8).is_some_and(|task| task == VAE_AUDIO_TASK))
        .collect::<Vec<_>>();
    assert_eq!(audio_rows.len(), 5);
    for fields in audio_rows {
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], "vae_architecture");
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], VAE_AUDIO_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-VAE-001");
        let contract_id = fields
            .first()
            .ok_or("audio VAE contract id is unavailable")?;
        let case_ids = audio_contract_case_ids
            .get(contract_id)
            .ok_or_else(|| format!("audio VAE contract {contract_id} has no fixture case"))?;
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join(&fields[2]))?)
            ),
            fields[5]
        );
        contract_evidence.push(json!({
            "contract_id": contract_id,
            "task_id": VAE_AUDIO_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": case_ids,
        }));
    }

    let structured_fixture_path =
        "crates/sim_test_support/fixtures/models/vae-structured/architecture-checkpoints.json";
    let structured_provenance_path =
        "crates/sim_test_support/fixtures/models/vae-structured/provenance.json";
    let structured_fixture_bytes = fs::read(workspace.join(structured_fixture_path))?;
    let structured_provenance_bytes = fs::read(workspace.join(structured_provenance_path))?;
    let structured_fixture: serde_json::Value = serde_json::from_slice(&structured_fixture_bytes)?;
    assert_eq!(structured_fixture["schema_version"], 1);
    assert_eq!(
        structured_fixture["fixture_id"],
        "sim-native-structured-vae-source-checkpoints-v1"
    );
    assert_eq!(
        structured_fixture["provenance_sha256"],
        format!("{:x}", Sha256::digest(&structured_provenance_bytes))
    );
    let structured_cases = structured_fixture["cases"]
        .as_array()
        .ok_or("structured VAE fixture cases are unavailable")?;
    assert_eq!(structured_cases.len(), 2);
    let mut structured_case_ids = BTreeSet::new();
    let mut structured_contract_case_ids = BTreeMap::<String, Vec<String>>::new();
    let mut structured_equation_checkpoint_count = 0_usize;
    let mut unique_structured_equation_checkpoints = BTreeSet::new();
    let mut structured_case_evidence = Vec::new();
    for case in structured_cases {
        let id = case["id"]
            .as_str()
            .ok_or("structured VAE fixture case id is unavailable")?;
        assert!(structured_case_ids.insert(id.to_owned()));
        let profile_name = case["profile"]
            .as_str()
            .ok_or("structured VAE fixture profile is unavailable")?;
        let profile = match profile_name {
            "HunyuanShapeV1" => VaeKernelProfile::HunyuanShapeV1,
            "TripoSplatV1" => VaeKernelProfile::TripoSplatV1,
            other => return Err(format!("unknown structured VAE fixture profile {other}").into()),
        };
        let plan = structured_vae_source_plan(&profile)?;
        assert_eq!(case["architecture"], plan.architecture());
        let equations = case["equation_checkpoints"]
            .as_array()
            .ok_or("structured VAE equation checkpoints are unavailable")?
            .iter()
            .map(|equation| {
                equation
                    .as_str()
                    .map(str::to_owned)
                    .ok_or("structured VAE equation checkpoint is unavailable")
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(equations, plan.equation_checkpoints());
        structured_equation_checkpoint_count += equations.len();
        unique_structured_equation_checkpoints.extend(equations.iter().cloned());
        let contract = case["catalog_contract_id"]
            .as_str()
            .ok_or("structured VAE catalog contract is unavailable")?;
        structured_contract_case_ids
            .entry(contract.to_owned())
            .or_default()
            .push(id.to_owned());
        structured_case_evidence.push(json!({
            "id": id,
            "profile": profile_name,
            "architecture": plan.architecture(),
            "structured_kind": case["structured_kind"],
            "source_state_count": structured_vae_source_state_count(&profile)?,
            "equation_checkpoints": equations,
            "native_encode_path": if profile == VaeKernelProfile::HunyuanShapeV1 {
                "typed_unavailable_pinned_source_missing_pre_kl_definition"
            } else {
                "not_exposed_by_source_decoder_contract"
            },
            "native_decode_path": "implemented",
            "validation_scope": "immutable source topology, exact state manifest, and named native semantic equation tests",
        }));
    }
    assert_eq!(structured_case_ids.len(), 2);
    assert_eq!(structured_contract_case_ids.len(), 2);
    assert_eq!(structured_equation_checkpoint_count, 15);
    assert_eq!(unique_structured_equation_checkpoints.len(), 15);
    let structured_rows = catalog_rows
        .iter()
        .filter(|fields| {
            fields
                .get(8)
                .is_some_and(|task| task == VAE_STRUCTURED_TASK)
        })
        .collect::<Vec<_>>();
    assert_eq!(structured_rows.len(), 2);
    for fields in structured_rows {
        assert_eq!(fields.len(), 15);
        assert_eq!(fields[1], "vae_architecture");
        assert_eq!(fields[7], "sim_model::vae");
        assert_eq!(fields[8], VAE_STRUCTURED_TASK);
        assert_eq!(fields[9], "VAL-VAE-001");
        assert_eq!(fields[10], "native_rust");
        assert_eq!(fields[14], "VAL-VAE-001");
        let contract_id = fields
            .first()
            .ok_or("structured VAE contract id is unavailable")?;
        let case_ids = structured_contract_case_ids
            .get(contract_id)
            .ok_or_else(|| format!("structured VAE contract {contract_id} has no fixture case"))?;
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(workspace.join(&fields[2]))?)
            ),
            fields[5]
        );
        contract_evidence.push(json!({
            "contract_id": contract_id,
            "task_id": VAE_STRUCTURED_TASK,
            "source_sha256": fields[5],
            "symbol_sha256": fields[6],
            "status": "passed",
            "case_ids": case_ids,
        }));
    }

    let implementation_path = "crates/sim_model/src/vae_architecture.rs";
    let implementation_bytes = fs::read(workspace.join(implementation_path))?;
    let implementation_sha256 = format!("{:x}", Sha256::digest(implementation_bytes));
    let producer_path = "crates/sim_model/tests/vae_architecture.rs";
    let producer_bytes = fs::read(workspace.join(producer_path))?;
    let producer_sha256 = format!("{:x}", Sha256::digest(producer_bytes));
    let tiling_implementations = [
        "crates/sim_model/src/vae.rs",
        "crates/sim_model/src/vae_tiling.rs",
        "crates/sim_model/tests/vae_architecture.rs",
        "crates/sim_model/tests/vae_tiling.rs",
    ]
    .into_iter()
    .map(|path| {
        Ok(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        }))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let image_implementations = [
        "crates/sim_model/src/vae.rs",
        "crates/sim_model/src/vae_architecture.rs",
        "crates/sim_model/src/vae_image.rs",
        "crates/sim_model/src/native_ops.rs",
        "crates/sim_runtime/src/assets.rs",
        "crates/sim_model/tests/vae_architecture.rs",
        "crates/sim_model/tests/vae_image.rs",
        "crates/sim_test_support/src/bin/generate_vae_image_fixture.rs",
        "crates/sim_test_support/tests/ownership_consolidation.rs",
    ]
    .into_iter()
    .map(|path| {
        Ok(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        }))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let video_implementations = [
        "crates/sim_model/src/vae.rs",
        "crates/sim_model/src/vae_architecture.rs",
        "crates/sim_model/src/vae_tiling.rs",
        "crates/sim_model/src/vae_video.rs",
        "crates/sim_model/src/native_ops.rs",
        "crates/sim_model/src/vision_models.rs",
        "crates/sim_runtime/src/assets.rs",
        "crates/sim_model/tests/vae_architecture.rs",
        "crates/sim_model/tests/vae_video.rs",
        "crates/sim_test_support/src/bin/generate_vae_video_fixture.rs",
        "crates/sim_test_support/tests/ownership_consolidation.rs",
    ]
    .into_iter()
    .map(|path| {
        Ok(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        }))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let audio_implementations = [
        "crates/sim_model/src/vae_architecture.rs",
        "crates/sim_model/src/vae_audio.rs",
        "crates/sim_model/src/vision_models.rs",
        "crates/sim_runtime/src/assets.rs",
        "crates/sim_model/tests/vae_architecture.rs",
        "crates/sim_model/tests/vae_audio.rs",
        "crates/sim_test_support/src/bin/generate_vae_audio_fixture.rs",
        "crates/sim_test_support/tests/ownership_consolidation.rs",
    ]
    .into_iter()
    .map(|path| {
        Ok(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        }))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let structured_implementations = [
        "crates/sim_model/src/sim_model.rs",
        "crates/sim_model/src/vae.rs",
        "crates/sim_model/src/vae_architecture.rs",
        "crates/sim_model/src/vae_structured.rs",
        "crates/sim_runtime/src/assets.rs",
        "crates/sim_model/tests/vae_architecture.rs",
        "crates/sim_model/tests/vae_structured.rs",
        "crates/sim_test_support/src/bin/generate_vae_structured_fixture.rs",
        "crates/sim_test_support/tests/ownership_consolidation.rs",
    ]
    .into_iter()
    .map(|path| {
        Ok(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        }))
    })
    .collect::<Result<Vec<_>, std::io::Error>>()?;
    let task_results = BTreeMap::from([
        (
            LOADER_TASK,
            json!({
                "status": "passed",
                "passed": loader_contract_count,
                "failed": 0,
                "skipped": 0,
                "implementation": {
                    "path": implementation_path,
                    "sha256": implementation_sha256,
                },
            }),
        ),
        (
            VAE_EXECUTION_TASK,
            json!({
                "status": "passed",
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "case_ids": [
                    "vae-execution:canonical-selection-binding",
                    "vae-execution:identity-round-trip",
                    "vae-execution:typed-boundary-rejection",
                ],
                "implementations": [{
                    "path": "crates/sim_model/src/vae.rs",
                    "sha256": format!(
                        "{:x}",
                        Sha256::digest(fs::read(workspace.join("crates/sim_model/src/vae.rs"))?)
                    ),
                }],
            }),
        ),
        (
            VAE_TILING_TASK,
            json!({
                "status": "passed",
                "passed": VAE_TILING_CONTRACTS.len(),
                "failed": 0,
                "skipped": 0,
                "case_ids": VAE_TILING_CASE_IDS,
                "implementations": tiling_implementations,
            }),
        ),
        (
            VAE_IMAGE_TASK,
            json!({
                "status": "passed",
                "passed": VAE_IMAGE_CONTRACTS.len(),
                "failed": 0,
                "skipped": 0,
                "case_ids": VAE_IMAGE_CASE_IDS,
                "fixture_case_ids": image_case_ids,
                "fixture_cases": image_case_evidence,
                "fixture_equation_checkpoint_count": image_equation_checkpoint_count,
                "fixture_unique_equation_checkpoint_count": unique_image_equation_checkpoints.len(),
                "fixture": {
                    "path": image_fixture_path,
                    "sha256": format!("{:x}", Sha256::digest(&image_fixture_bytes)),
                    "provenance_path": image_provenance_path,
                    "provenance_sha256": format!("{:x}", Sha256::digest(&image_provenance_bytes)),
                },
                "implementations": image_implementations,
            }),
        ),
        (
            VAE_VIDEO_TASK,
            json!({
                "status": "passed",
                "passed": video_contract_case_ids.len(),
                "failed": 0,
                "skipped": 0,
                "fixture_case_ids": video_case_ids,
                "fixture_cases": video_case_evidence,
                "fixture_equation_checkpoint_count": video_equation_checkpoint_count,
                "fixture_unique_equation_checkpoint_count": unique_video_equation_checkpoints.len(),
                "semantic_test_suites": [
                    "sim_model::vae_video::tests",
                    "sim_model/tests/vae_video.rs"
                ],
                "fixture": {
                    "path": video_fixture_path,
                    "sha256": format!("{:x}", Sha256::digest(&video_fixture_bytes)),
                    "provenance_path": video_provenance_path,
                    "provenance_sha256": format!("{:x}", Sha256::digest(&video_provenance_bytes)),
                },
                "implementations": video_implementations,
            }),
        ),
        (
            VAE_AUDIO_TASK,
            json!({
                "status": "passed",
                "passed": audio_contract_case_ids.len(),
                "failed": 0,
                "skipped": 0,
                "fixture_case_ids": audio_case_ids,
                "fixture_cases": audio_case_evidence,
                "fixture_equation_checkpoint_count": audio_equation_checkpoint_count,
                "fixture_unique_equation_checkpoint_count": unique_audio_equation_checkpoints.len(),
                "semantic_test_suites": [
                    "sim_model::vae_audio::tests",
                    "sim_model/tests/vae_audio.rs",
                    "sim_runtime::assets::tests::audio_vae_production_admission_is_authorized_canonical_and_failure_atomic"
                ],
                "fixture": {
                    "path": audio_fixture_path,
                    "sha256": format!("{:x}", Sha256::digest(&audio_fixture_bytes)),
                    "provenance_path": audio_provenance_path,
                    "provenance_sha256": format!("{:x}", Sha256::digest(&audio_provenance_bytes)),
                },
                "implementations": audio_implementations,
            }),
        ),
        (
            VAE_STRUCTURED_TASK,
            json!({
                "status": "passed",
                "passed": structured_contract_case_ids.len(),
                "failed": 0,
                "skipped": 0,
                "fixture_case_ids": structured_case_ids,
                "fixture_cases": structured_case_evidence,
                "fixture_equation_checkpoint_count": structured_equation_checkpoint_count,
                "fixture_unique_equation_checkpoint_count": unique_structured_equation_checkpoints.len(),
                "semantic_test_suites": [
                    "sim_model::vae_structured::tests",
                    "sim_model/tests/vae_structured.rs",
                    "sim_runtime::assets::AssetService::load_structured_vae_with_context"
                ],
                "fixture": {
                    "path": structured_fixture_path,
                    "sha256": format!("{:x}", Sha256::digest(&structured_fixture_bytes)),
                    "provenance_path": structured_provenance_path,
                    "provenance_sha256": format!("{:x}", Sha256::digest(&structured_provenance_bytes)),
                },
                "implementations": structured_implementations,
            }),
        ),
    ]);
    let artifact = json!({
        "schema_version": 1,
        "validation_id": "VAL-VAE-001",
        "overall_status": "partial",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": {
            "passed": contract_evidence.len(),
            "failed": 0,
            "skipped": 0,
        },
        "implementation": {
            "path": producer_path,
            "sha256": producer_sha256,
        },
        "task_results": task_results,
        "contracts": contract_evidence,
        "selector_cases": validated,
        "remaining_tasks": [
            "comfy-parity-vae-owner-consolidation"
        ],
    });
    let directory = workspace.join("target/sim-parity");
    fs::create_dir_all(&directory)?;
    let temporary = directory.join("val-vae-001.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&artifact)?)?;
    fs::rename(temporary, directory.join("val-vae-001.json"))?;
    Ok(())
}

#[test]
fn nested_profile_decisions_are_deterministic() -> Result<(), Box<dyn Error>> {
    let registry = VaeArchitectureRegistry::checked()?;
    let cancellation = CancellationToken::default();

    for (channels, version, extra) in [(512, 0_u8, false), (1024, 1, false), (1024, 2, true)] {
        let mut fixture = SelectorFixture::new(
            "conditioning-vae-selection-sd-l651-b808b0d3",
            vec![(
                "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
                vec![channels, 128, 3, 3, 3],
            )],
        );
        if extra {
            fixture
                .tensors
                .push(("encoder.down_blocks.1.conv.conv.bias", vec![1], "F32"));
        }
        fixture.metadata.insert(
            "config".to_owned(),
            r#"{"vae":{"causal":true,"version":"fixture"}}"#.to_owned(),
        );
        let selection = registry.select(&fixture.probe()?, &cancellation)?;
        assert!(matches!(
            selection.loader_configuration(),
            VaeLoaderConfiguration::LtxVideo {
                configuration_sha256: Some(digest),
                configuration_json: Some(configuration),
            } if digest.len() == 64 && configuration.contains("\"causal\":true")
        ));
        assert!(match (version, selection.profile()) {
            (
                0,
                VaeKernelProfile::LtxVideoV0 {
                    configuration_sha256,
                },
            )
            | (
                1,
                VaeKernelProfile::LtxVideoV1 {
                    configuration_sha256,
                },
            )
            | (
                2,
                VaeKernelProfile::LtxVideoV2 {
                    configuration_sha256,
                },
            ) => {
                configuration_sha256
                    .as_deref()
                    .is_some_and(|digest| digest.len() == 64)
            }
            _ => false,
        });
    }

    let oobleck_44 = SelectorFixture::new(
        "conditioning-vae-selection-sd-l609-82ce1c2f",
        vec![("decoder.layers.1.layers.0.beta", vec![64])],
    );
    assert_eq!(
        registry
            .select(&oobleck_44.probe()?, &cancellation)?
            .profile(),
        &VaeKernelProfile::AudioOobleck44KhzV1
    );
    let oobleck_48 = SelectorFixture::new(
        "conditioning-vae-selection-sd-l609-82ce1c2f",
        vec![
            ("decoder.layers.1.layers.0.beta", vec![64]),
            ("decoder.layers.2.layers.1.weight_v", vec![64, 12]),
        ],
    );
    assert_eq!(
        registry
            .select(&oobleck_48.probe()?, &cancellation)?
            .profile(),
        &VaeKernelProfile::AudioOobleck48KhzV1
    );
    let mochi = registry.select(
        &SelectorFixture::new(
            "conditioning-vae-selection-sd-l636-d758fbe2",
            vec![("blocks.2.blocks.3.stack.5.weight", vec![1])],
        )
        .probe()?,
        &cancellation,
    )?;
    assert_eq!(mochi.supported_dtypes(), &[DType::F16, DType::F32]);
    assert!(!mochi.supported_dtypes().contains(&DType::Bf16));

    let shallow = SelectorFixture::new(
        "conditioning-vae-selection-sd-l874-13e203b0",
        vec![("decoder.layers.3.transformers.0.pre_norm.alpha", vec![1])],
    );
    let mut deep = shallow.clone();
    deep.tensors.push((
        "decoder.layers.3.transformers.11.self_attn.to_out.weight",
        vec![1],
        "F32",
    ));
    assert_eq!(
        registry.select(&shallow.probe()?, &cancellation)?.profile(),
        &VaeKernelProfile::StableAudio3ShallowV1
    );
    assert_eq!(
        registry.select(&deep.probe()?, &cancellation)?.profile(),
        &VaeKernelProfile::StableAudio3DeepV1
    );
    Ok(())
}

#[test]
fn val_vae_001_source_loader_configuration_and_conversion_are_checked() -> Result<(), Box<dyn Error>>
{
    let registry = VaeArchitectureRegistry::checked()?;
    let cancellation = CancellationToken::default();

    let temporal = registry.select(
        &probe(
            &[("decoder.mid.block_1.mix_factor", Vec::new(), "F32")],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(temporal.latent_dimensions(), 2);
    assert_eq!(temporal.boundary(), sim_model::VaeBoundaryDomain::Image);

    let mut taesd_metadata = BTreeMap::new();
    taesd_metadata.insert("tae_latent_channels".to_owned(), "4".to_owned());
    let taesd = registry.select(
        &probe(
            &[("taesd_decoder.1.weight", vec![1, 8], "F32")],
            taesd_metadata,
        )?,
        &cancellation,
    )?;
    assert!(matches!(
        taesd.loader_configuration(),
        VaeLoaderConfiguration::Taesd {
            latent_channels: 4,
            metadata_override: true
        }
    ));

    let default = registry.select(
        &probe(
            &[
                ("decoder.conv_in.weight", vec![256, 4, 3, 3], "F32"),
                ("decoder.post_quant_conv.weight", vec![4, 4, 1, 1], "F32"),
                ("bn.running_mean", vec![4], "F32"),
            ],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert!(matches!(
        default.loader_configuration(),
        VaeLoaderConfiguration::DefaultKl {
            x4: true,
            legacy_prefix_rewrite: true,
            batch_norm_latent: true,
            asymmetric_decoder_channels: Some(64),
            embed_dim: Some(4),
        }
    ));
    assert_eq!(default.latent_channels(), Some(16));
    assert_eq!(
        default.profile(),
        &VaeKernelProfile::AutoencoderKlBatchNormV1
    );
    assert_eq!(
        default.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["Flux", "SD3"])
    );

    let x4 = registry.select(
        &probe(
            &[("decoder.conv_in.weight", vec![128, 4, 3, 3], "F32")],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(x4.profile(), &VaeKernelProfile::AutoencodingEngineX4V1);
    assert_eq!(
        x4.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["SD_X4"])
    );

    let standard = registry.select(
        &probe(
            &[
                ("decoder.conv_in.weight", vec![128, 4, 3, 3], "F32"),
                (
                    "encoder.down.2.downsample.conv.weight",
                    vec![128, 128, 3, 3],
                    "F32",
                ),
            ],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(standard.profile(), &VaeKernelProfile::AutoencodingEngineV1);

    let diffusers = registry.select(
        &probe(
            &[
                (
                    "decoder.up_blocks.0.resnets.0.norm1.weight",
                    vec![128],
                    "F32",
                ),
                ("decoder.conv_in.weight", vec![128, 4, 3, 3], "F32"),
                (
                    "decoder.mid_block.attentions.0.to_q.weight",
                    vec![128, 128],
                    "F32",
                ),
            ],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert!(matches!(
        diffusers.loader_configuration(),
        VaeLoaderConfiguration::DiffusersPreconverted {
            key_mapping,
            conv3d: false,
            inner,
        } if key_mapping.len() == 3
            && key_mapping.iter().any(|(source, target)|
                source == "decoder.up_blocks.0.resnets.0.norm1.weight"
                    && target == "decoder.up.3.block.0.norm1.weight")
            && matches!(inner.as_ref(), VaeLoaderConfiguration::DefaultKl { .. })
    ));
    assert!(diffusers.trace().contains(&VAE_DIFFUSERS_ROW_ID));

    let malformed = probe(
        &[
            ("decoder.up_blocks.0.resnets.0.norm1.weight", vec![0], "F32"),
            ("decoder.conv_in.weight", vec![128, 4, 3, 3], "F32"),
        ],
        BTreeMap::new(),
    )?;
    assert!(matches!(
        registry.select(&malformed, &cancellation),
        Err(VaeArchitectureError::MalformedDiffusersTensor { .. })
    ));

    let collision = probe(
        &[
            (
                "decoder.up_blocks.0.resnets.0.norm1.weight",
                vec![128],
                "F32",
            ),
            ("decoder.up.3.block.0.norm1.weight", vec![128], "F32"),
        ],
        BTreeMap::new(),
    )?;
    assert!(matches!(
        registry.select(&collision, &cancellation),
        Err(VaeArchitectureError::DiffusersKeyCollision { .. })
    ));

    let explicit = registry.select_explicit(
        &probe(&[], BTreeMap::new())?,
        r#"{"params":{"embed_dim":4,"ddconfig":{"attn_resolutions":[],"ch":32,"ch_mult":[1],"double_z":true,"in_channels":3,"num_res_blocks":1,"out_ch":3,"resolution":8,"z_channels":4}}}"#,
        &cancellation,
    )?;
    assert!(matches!(
        explicit.loader_configuration(),
        VaeLoaderConfiguration::ExplicitAutoencoderKl { params_sha256, params_json }
            if params_sha256.len() == 64 && params_json.contains("\"embed_dim\":4")
    ));
    assert!(matches!(
        registry.select_explicit(
            &probe(&[], BTreeMap::new())?,
            r#"{"params":null}"#,
            &cancellation,
        ),
        Err(VaeArchitectureError::MalformedConfiguration(_))
    ));
    assert!(matches!(
        registry.select_explicit(
            &probe(&[], BTreeMap::new())?,
            r#"{"params":{"embed_dim":4,"ddconfig":{"attn_resolutions":[],"ch":32,"ch_mult":[1],"double_z":true,"in_channels":3,"num_res_blocks":1,"out_ch":3,"resolution":8,"z_channels":4},"decoder_ddconfig":{"attn_resolutions":[],"ch":32,"ch_mult":[1],"double_z":true,"in_channels":3,"num_res_blocks":1,"out_ch":3,"resolution":8,"use_linear_attn":true,"z_channels":4}}}"#,
            &cancellation,
        ),
        Err(VaeArchitectureError::MalformedConfiguration(message))
            if message.contains("decoder_ddconfig") && message.contains("use_linear_attn=false")
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        registry.select(&malformed, &cancelled),
        Err(VaeArchitectureError::Cancelled(_))
    ));
    assert!(matches!(
        registry.select_explicit(
            &probe(&[], BTreeMap::new())?,
            r#"{"params":{}}"#,
            &cancelled,
        ),
        Err(VaeArchitectureError::Cancelled(_))
    ));
    Ok(())
}

#[test]
fn taesd_128_flux2_target_is_reachable() -> Result<(), Box<dyn Error>> {
    let registry = VaeArchitectureRegistry::checked()?;
    let cancellation = CancellationToken::default();
    let selection = registry.select(
        &probe(
            &[("taesd_decoder.1.weight", vec![1, 128], "F32")],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(selection.latent_channels(), Some(128));
    assert_eq!(selection.target_latent_channels(), Some(128));
    assert_eq!(
        selection.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["SD15", "SDXL", "SD_X4", "Flux2"])
    );
    selection.ensure_native_builder_available()?;

    let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
    let family = family_registry
        .definitions_in_source_order()
        .into_iter()
        .find(|definition| definition.identifier == "Flux2")
        .ok_or("canonical Flux2 family is unavailable")?;
    let target = VaeExecutionTarget::new(
        ModelFamilyIdentity::new(
            family.feature_id,
            family.identifier,
            family.architecture_version,
        )?,
        LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
        DType::F32,
        DeviceId::CPU,
    );
    registry.validate_target(
        &selection,
        &target,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    let intended = registry.intended_target(
        &selection,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    assert_eq!(intended.latent_format().identifier(), "Flux2");
    Ok(())
}

#[test]
fn corrected_nested_target_compatibility_is_exact() -> Result<(), Box<dyn Error>> {
    let registry = VaeArchitectureRegistry::checked()?;
    let cancellation = CancellationToken::default();
    let refiner = registry.select(
        &probe(
            &[("decoder.conv_in.weight", vec![128, 32, 3, 3, 3], "F32")],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(refiner.latent_channels(), Some(32));
    assert_eq!(refiner.target_latent_channels(), Some(64));
    assert_eq!(
        refiner.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["HunyuanImage21Refiner"])
    );
    let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
    let refiner_target =
        registry.intended_target(&refiner, &family_registry, &latent_registry, &cancellation)?;
    assert_eq!(
        refiner_target.family().identifier(),
        "HunyuanImage21Refiner"
    );
    assert_eq!(
        refiner_target.latent_format().identifier(),
        "HunyuanImage21Refiner"
    );
    registry.validate_target(
        &refiner,
        &refiner_target,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;

    let music = registry.select(
        &probe(
            &[(
                "vocoder.backbone.channel_layers.0.0.bias",
                Vec::new(),
                "F32",
            )],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert_eq!(
        music.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["ACEAudio"])
    );

    let mut audio_metadata = BTreeMap::new();
    audio_metadata.insert(
        "config".to_owned(),
        r#"{"audio_vae":{"model":{"params":{"ddconfig":{"z_channels":8},"sampling_rate":16000}},"preprocessing":{"stft":{"hop_length":160}}},"vocoder":{"upsample_rates":[5,4,2,2,2]}}"#.to_owned(),
    );
    let ltx_audio = registry.select(
        &probe(
            &[("vocoder.resblocks.0.convs1.0.weight", vec![1], "F32")],
            audio_metadata.clone(),
        )?,
        &cancellation,
    )?;
    assert_eq!(
        ltx_audio.canonical_compatibility(),
        VaeCanonicalCompatibility::Exact(&["ACEAudio"])
    );
    let mut reordered_audio_metadata = BTreeMap::new();
    reordered_audio_metadata.insert(
        "config".to_owned(),
        r#"{"vocoder":{"upsample_rates":[5,4,2,2,2]},"audio_vae":{"preprocessing":{"stft":{"hop_length":160}},"model":{"params":{"sampling_rate":16000,"ddconfig":{"z_channels":8}}}}}"#.to_owned(),
    );
    let reordered_audio = registry.select(
        &probe(
            &[("vocoder.resblocks.0.convs1.0.weight", vec![1], "F32")],
            reordered_audio_metadata,
        )?,
        &cancellation,
    )?;
    assert_eq!(
        ltx_audio.loader_configuration(),
        reordered_audio.loader_configuration()
    );
    assert_eq!(ltx_audio.supported_dtypes(), &[DType::F32]);
    assert_eq!(ltx_audio.latent_channels(), Some(8));
    assert!(matches!(
        ltx_audio.loader_configuration(),
        VaeLoaderConfiguration::LtxAudio {
            autoencoder_sha256,
            autoencoder_json,
            vocoder_sha256,
            vocoder_json,
            latent_channels: 8,
            input_sample_rate: 16_000,
            output_sample_rate: 16_000,
        } if autoencoder_sha256.len() == 64
            && vocoder_sha256.len() == 64
            && autoencoder_json.contains("\"z_channels\":8")
            && vocoder_json.contains("\"upsample_rates\"")
    ));
    let missing_config = probe(
        &[("vocoder.resblocks.0.convs1.0.weight", vec![1], "F32")],
        BTreeMap::new(),
    )?;
    assert!(matches!(
        registry.select(&missing_config, &cancellation),
        Err(VaeArchitectureError::MalformedMetadata { .. })
    ));
    let family_registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let latent_registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
    let ltx_audio_target = registry.intended_target(
        &ltx_audio,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    assert_eq!(ltx_audio_target.latent_format().identifier(), "ACEAudio");
    let mut bad_numeric_metadata = BTreeMap::new();
    bad_numeric_metadata.insert(
        "config".to_owned(),
        r#"{"audio_vae":{"model":{"params":{"ddconfig":{"z_channels":"16"}}}},"vocoder":{}}"#
            .to_owned(),
    );
    assert!(matches!(
        registry.select(
            &probe(
                &[("vocoder.resblocks.0.convs1.0.weight", vec![1], "F32")],
                bad_numeric_metadata,
            )?,
            &cancellation,
        ),
        Err(VaeArchitectureError::MalformedMetadata { .. })
    ));

    let fallback = registry.select(
        &probe(
            &[(
                "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
                vec![768, 128, 3, 3, 3],
                "F32",
            )],
            BTreeMap::new(),
        )?,
        &cancellation,
    )?;
    assert!(matches!(
        fallback.profile(),
        VaeKernelProfile::LtxVideoV0 { .. }
    ));
    Ok(())
}

#[test]
fn canonical_target_validation_and_model_store_probe_are_exact() -> Result<(), Box<dyn Error>> {
    let registry = VaeArchitectureRegistry::checked()?;
    let family_registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let latent_registry = LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)?;
    let family = family_registry
        .definitions_in_source_order()
        .into_iter()
        .find(|definition| definition.identifier == "CogVideoX_T2V")
        .ok_or("canonical CogVideoX family is unavailable")?;
    let target = VaeExecutionTarget::new(
        ModelFamilyIdentity::new(
            family.feature_id,
            family.identifier,
            family.architecture_version,
        )?,
        LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)?,
        DType::F32,
        DeviceId::CPU,
    );
    let fixture = selector_fixtures()
        .into_iter()
        .find(|fixture| fixture.row == "conditioning-vae-selection-sd-l690-3abc5b4d")
        .ok_or("CogVideoX VAE fixture is unavailable")?;
    let cancellation = CancellationToken::default();
    let selection = registry.select_for_target(
        &fixture.probe()?,
        &target,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    assert_eq!(selection.profile(), &VaeKernelProfile::CogVideoXV1);

    let artifact = ArtifactRecord {
        key: ArtifactKey::new("models", "cog-vae.safetensors")?,
        namespace: "vae".to_owned(),
        canonical_path: PathBuf::from("/verified/models/cog-vae.safetensors"),
        byte_size: 1,
        modified_nanoseconds: 1,
        sha256: "a".repeat(64),
        availability: ArtifactAvailability::Present,
    };
    let patch = PatchGraph::checked_semantic(artifact.sha256.clone(), Vec::new())?.identity();
    let descriptor = VaeDescriptor::checked_selection(
        &artifact,
        &selection,
        &target,
        &family_registry,
        &latent_registry,
        patch.clone(),
        VaeBoundary::video(3)?,
        [0.0, 1.0],
        &cancellation,
    )?;
    assert_eq!(descriptor.identity().family(), target.family());
    assert_eq!(
        descriptor.identity().latent_format(),
        target.latent_format()
    );
    assert_eq!(
        descriptor.identity().architecture(),
        selection.architecture()
    );
    assert_eq!(descriptor.identity().profile(), selection.profile());
    assert_eq!(
        descriptor.identity().loader_configuration(),
        selection.loader_configuration()
    );
    let encoded_identity = serde_json::to_value(descriptor.identity())?;
    let mut wrong_loader_identity = encoded_identity.clone();
    wrong_loader_identity["loader_configuration"] = json!({
        "kind": "explicit_autoencoder_kl",
        "params_sha256": "a".repeat(64),
        "params_json": "{}"
    });
    let error = serde_json::from_value::<sim_model::VaeIdentity>(wrong_loader_identity)
        .expect_err("retained loader configuration must remain canonical and digest-bound");
    assert!(error.to_string().contains("digest-bound"));
    let mut wrong_family_identity = encoded_identity.clone();
    wrong_family_identity["family"]["identifier"] = json!("unknown_valid_family");
    let error = serde_json::from_value::<sim_model::VaeIdentity>(wrong_family_identity)
        .expect_err("unknown canonical family must fail during identity deserialization");
    assert!(error.to_string().contains("unknown canonical model family"));
    let mut wrong_latent_identity = encoded_identity;
    wrong_latent_identity["latent_format"] = json!({
        "schema_version": 1,
        "feature_id": "COMFY-MODEL-0028",
        "identifier": "Cosmos1CV8x8x8"
    });
    let error = serde_json::from_value::<sim_model::VaeIdentity>(wrong_latent_identity)
        .expect_err("wrong canonical latent must fail during identity deserialization");
    assert!(error.to_string().contains("requires latent"));
    assert!(matches!(
        VaeDescriptor::checked_selection(
            &artifact,
            &selection,
            &target,
            &family_registry,
            &latent_registry,
            patch,
            VaeBoundary::image(3)?,
            [0.0, 1.0],
            &cancellation,
        ),
        Err(VaeError::SelectionBoundaryMismatch { .. })
    ));

    let wrong_latent = VaeExecutionTarget::new(
        target.family().clone(),
        LatentFormatIdentity::new("COMFY-MODEL-0028", "Cosmos1CV8x8x8")?,
        DType::F32,
        DeviceId::CPU,
    );
    assert!(matches!(
        registry.validate_target(
            &selection,
            &wrong_latent,
            &family_registry,
            &latent_registry,
            &cancellation,
        ),
        Err(VaeArchitectureError::FamilyLatentMismatch { .. })
    ));

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("cog-vae.safetensors");
    write_safetensors(&path, &fixture.tensors)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "models",
        "vae",
        directory.path(),
        ["safetensors"],
    )?)?;
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(
        &index,
        &ArtifactKey::new("models", "cog-vae.safetensors")?,
        &cancellation,
    )?;
    assert_eq!(loaded.accounting().resident_bytes, 0);
    let selected = registry.select_loaded(
        &store,
        &loaded,
        &target,
        &family_registry,
        &latent_registry,
        &cancellation,
    )?;
    assert_eq!(selected.profile(), &VaeKernelProfile::CogVideoXV1);
    assert_eq!(loaded.accounting().resident_bytes, 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        registry.select(&fixture.probe()?, &cancelled),
        Err(VaeArchitectureError::Cancelled(_))
    ));
    Ok(())
}

fn write_safetensors(
    path: &Path,
    tensors: &[(&str, Vec<u64>, &str)],
) -> Result<(), Box<dyn Error>> {
    let mut header = serde_json::Map::new();
    let mut offset = 0_u64;
    let mut payload = Vec::new();
    for (name, shape, dtype) in tensors {
        let element_bytes = match *dtype {
            "F16" | "BF16" => 2_u64,
            "F32" => 4,
            other => return Err(format!("unsupported test dtype {other}").into()),
        };
        let elements = shape
            .iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or("test tensor shape overflow")?;
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or("test tensor byte overflow")?;
        let end = offset
            .checked_add(bytes)
            .ok_or("test tensor offset overflow")?;
        header.insert(
            (*name).to_owned(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}),
        );
        payload.resize(usize::try_from(end)?, 0);
        offset = end;
    }
    let header = serde_json::to_vec(&header)?;
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&u64::try_from(header.len())?.to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes)?;
    Ok(())
}

fn parse_catalog(encoded: &str) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
    let mut rows = Vec::new();
    for line in encoded.lines().skip(1) {
        let mut fields = Vec::new();
        let mut field = String::new();
        let mut characters = line.chars().peekable();
        let mut quoted = false;
        while let Some(character) = characters.next() {
            match character {
                '"' if quoted && characters.peek() == Some(&'"') => {
                    field.push('"');
                    characters.next();
                }
                '"' => quoted = !quoted,
                ',' if !quoted => fields.push(std::mem::take(&mut field)),
                _ => field.push(character),
            }
        }
        if quoted {
            return Err("unterminated CSV quoted field".into());
        }
        fields.push(field);
        rows.push(fields);
    }
    Ok(rows)
}
