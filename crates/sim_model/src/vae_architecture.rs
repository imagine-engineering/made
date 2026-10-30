use crate::{
    GENERATED_LATENT_FORMATS, GENERATED_MODEL_FAMILY_REGISTRATIONS, LatentFormatIdentity,
    LatentFormatRegistry, LoadedModel, ModelFamilyIdentity, ModelFamilyRegistry, ModelProbe,
    ModelStorageDType, ModelStore, ModelStoreError, VaeArchitectureIdentity,
    VaeCanonicalCompatibility, VaeKernelProfile,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{DType, DeviceId};
use sim_types::{CancellationError, CancellationToken, DeviceKind};
use std::sync::Arc;
use thiserror::Error;

pub const VAE_SELECTOR_SOURCE_PATH: &str = "projects/comfy/ComfyUI/sim/sd.py";
pub const VAE_SELECTOR_SOURCE_SHA256: &str =
    "9c378edbcaab01d00397cc0ef1cab7d37d25fca49c707a87cde451030ac6bf42";
pub const VAE_SELECTOR_ROW_COUNT: usize = 32;
pub const VAE_SELECTOR_BRANCH_COUNT: usize = 30;
pub const VAE_DIFFUSERS_ROW_ID: &str = "conditioning-vae-state-dict-conversion-sd-l473-62acfe89";
pub const VAE_AUTOMATIC_ROW_ID: &str = "conditioning-vae-selection-sd-l503-9b4d93ce";
pub const VAE_UNBOUND_ROW_ID: &str = "conditioning-vae-architecture-sd-unbound-9e179c50";
pub const VAE_DIFFUSERS_SENTINEL: &str = "decoder.up_blocks.0.resnets.0.norm1.weight";
const MAX_VAE_CONFIGURATION_BYTES: usize = 256 * 1024;
const MODEL_PROBE_DTYPE_PREFIX: &str = "__sim.model_probe.v1.dtype.";
const IMAGE_NATIVE_DEVICES: &[DeviceKind] = &[DeviceKind::Cpu];

const L504: &str = "conditioning-vae-selection-sd-l504-48d322ed";
const L512: &str = "conditioning-vae-selection-sd-l512-43630fab";
const L518: &str = "conditioning-vae-selection-sd-l518-e78759a4";
const L527: &str = "conditioning-vae-selection-sd-l527-8575102a";
const L535: &str = "conditioning-vae-selection-sd-l535-1ad96396";
const L542: &str = "conditioning-vae-selection-sd-l542-f7074db0";
const L546: &str = "conditioning-vae-selection-sd-l546-cf72d403";
const L547: &str = "conditioning-vae-selection-sd-l547-3d4531f9";
const L559: &str = "conditioning-vae-selection-sd-l559-295b67db";
const L603: &str = "conditioning-vae-selection-sd-l603-86be7303";
const L609: &str = "conditioning-vae-selection-sd-l609-82ce1c2f";
const L636: &str = "conditioning-vae-selection-sd-l636-d758fbe2";
const L651: &str = "conditioning-vae-selection-sd-l651-b808b0d3";
const L673: &str = "conditioning-vae-selection-sd-l673-58c13833";
const L690: &str = "conditioning-vae-selection-sd-l690-3abc5b4d";
const L701: &str = "conditioning-vae-selection-sd-l701-7056d652";
const L717: &str = "conditioning-vae-selection-sd-l717-2e6a172a";
const L730: &str = "conditioning-vae-selection-sd-l730-7d7bc483";
const L731: &str = "conditioning-vae-selection-sd-l731-01dbf62f";
const L762: &str = "conditioning-vae-selection-sd-l762-d9946728";
const L784: &str = "conditioning-vae-selection-sd-l784-3c01e5be";
const L799: &str = "conditioning-vae-selection-sd-l799-186f31a5";
const L809: &str = "conditioning-vae-selection-sd-l809-83aeacf2";
const L828: &str = "conditioning-vae-selection-sd-l828-58c549be";
const L835: &str = "conditioning-vae-selection-sd-l835-670a25d1";
const L840: &str = "conditioning-vae-selection-sd-l840-410d45c2";
const L856: &str = "conditioning-vae-selection-sd-l856-a443d5ce";
const L874: &str = "conditioning-vae-selection-sd-l874-13e203b0";
const L902: &str = "conditioning-vae-selection-sd-l902-048f369c";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VaeCatalogRowKind {
    StateDictConversion,
    ArchitectureUnavailable,
    SelectionBranch,
}

impl VaeCatalogRowKind {
    pub const fn catalog_name(self) -> &'static str {
        match self {
            Self::StateDictConversion => "vae_state_dict_conversion",
            Self::ArchitectureUnavailable => "vae_architecture",
            Self::SelectionBranch => "vae_selection_branch",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VaeSelectorCatalogRow {
    pub contract_id: &'static str,
    pub kind: VaeCatalogRowKind,
    pub source_symbol: &'static str,
    pub source_ordinal: u16,
    pub symbol_sha256: &'static str,
}

macro_rules! row {
    ($id:expr, $kind:ident, $symbol:expr, $ordinal:expr, $digest:expr) => {
        VaeSelectorCatalogRow {
            contract_id: $id,
            kind: VaeCatalogRowKind::$kind,
            source_symbol: $symbol,
            source_ordinal: $ordinal,
            symbol_sha256: $digest,
        }
    };
}

pub const VAE_SELECTOR_CATALOG_ROWS: &[VaeSelectorCatalogRow] = &[
    row!(
        VAE_DIFFUSERS_ROW_ID,
        StateDictConversion,
        "VAE.__init__.state_dict_conversion@L473:'decoder.up_blocks.0.resnets.0.norm1.weight' in sd.keys()",
        27,
        "8a40e47b9f44b53f57dc4b8f2ef4317fbb0c12e930f71c0b1585a7971c42d692"
    ),
    row!(
        VAE_UNBOUND_ROW_ID,
        ArchitectureUnavailable,
        "VAE.__init__.unbound@L914",
        57,
        "7ae0c744d38099d04600831b3b58ebed9a369c4d75585368893eb925795f9f27"
    ),
    row!(
        VAE_AUTOMATIC_ROW_ID,
        SelectionBranch,
        "VAE.__init__.selection@L503:config is None",
        59,
        "96f8ac9af70dd4772507a30a1b81cd88b5e16458901b45815ca2fc11d4f08514"
    ),
    row!(
        L504,
        SelectionBranch,
        r#"VAE.__init__.selection@L504:"decoder.mid.block_1.mix_factor" in sd"#,
        60,
        "2ec55e3530257257988d2ba6f55f5317b5b580cf2b5e91a811321532b10401f3"
    ),
    row!(
        L512,
        SelectionBranch,
        r#"VAE.__init__.selection@L512:"taesd_decoder.1.weight" in sd"#,
        61,
        "313a317ed34e9c0a152e0ef61f2f6da1e3e58007f3c709bd4645ac05151ce644"
    ),
    row!(
        L518,
        SelectionBranch,
        r#"VAE.__init__.selection@L518:"vquantizer.codebook.weight" in sd"#,
        62,
        "404e2d7603a30519fef3e5a449cfb49e5222b6140982484b5b85490fc6448f70"
    ),
    row!(
        L527,
        SelectionBranch,
        r#"VAE.__init__.selection@L527:"backbone.1.0.block.0.1.num_batches_tracked" in sd"#,
        63,
        "6346cfddc115755abbbf49be7f73e656fd770a1ecf5278a3b17a6f4341588fba"
    ),
    row!(
        L535,
        SelectionBranch,
        r#"VAE.__init__.selection@L535:"blocks.11.num_batches_tracked" in sd"#,
        64,
        "424874d66f094988abb1c00d7d00f32566a816f31b3638efb756c093c71360ba"
    ),
    row!(
        L542,
        SelectionBranch,
        r#"VAE.__init__.selection@L542:"encoder.backbone.1.0.block.0.1.num_batches_tracked" in sd"#,
        65,
        "b88c660f2a082b89c4afb78bac69a9550a0f218e28714cd231098342b062a030"
    ),
    row!(
        L546,
        SelectionBranch,
        r#"VAE.__init__.selection@L546:"decoder.conv_in.weight" in sd"#,
        66,
        "6b9d0d43cd1877845c8556e4f1181fbc7218e4065536a6adb2403af600577058"
    ),
    row!(
        L547,
        SelectionBranch,
        "VAE.__init__.selection@L547:sd['decoder.conv_in.weight'].shape[1] == 64",
        67,
        "d4483ebc0bc19b02f7160a6bc56378bb844e88313f17de14de8f94323006f0e4"
    ),
    row!(
        L559,
        SelectionBranch,
        "VAE.__init__.selection@L559:sd['decoder.conv_in.weight'].shape[1] == 32 and sd['decoder.conv_in.weight'].ndim == 5",
        68,
        "ab8586928e995f729e9b8e5a384fc0edfeb985715f40ab8f4482b155f953f454"
    ),
    row!(
        L603,
        SelectionBranch,
        "VAE.__init__.selection@L603:'post_quant_conv.weight' in sd",
        69,
        "d910eb47f800a18ef321590e88243ff073b1a164e87c65bb972e9a61b4af8cce"
    ),
    row!(
        L609,
        SelectionBranch,
        r#"VAE.__init__.selection@L609:"decoder.layers.1.layers.0.beta" in sd"#,
        70,
        "ad6fa0aab95ffa5d7adc60f3680924f35691b5752c35528062364f84de571dab"
    ),
    row!(
        L636,
        SelectionBranch,
        r#"VAE.__init__.selection@L636:"blocks.2.blocks.3.stack.5.weight" in sd or "decoder.blocks.2.blocks.3.stack.5.weight" in sd or "layers.4.layers.1.attn_block.attn.qkv.weight" in sd or "encoder.layers.4.layers.1.attn_block.attn.qkv.weight" in sd"#,
        71,
        "b5c02f4d0527117cf42e0885df663aaaf4692d3da88af14ce90577bf7eb25c82"
    ),
    row!(
        L651,
        SelectionBranch,
        r#"VAE.__init__.selection@L651:"decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in sd"#,
        72,
        "6d7d7bd03e5bcfb52bd589029264e2e4f01b3dd0a63feac1063ddfda1f0492e7"
    ),
    row!(
        L673,
        SelectionBranch,
        r#"VAE.__init__.selection@L673:"decoder.conv_in.conv.weight" in sd and sd['decoder.conv_in.conv.weight'].shape[1] == 32"#,
        73,
        "76e52d5641dad82bcb7cb832fe2949e3c6e7c27f850b34346ad2fca4eb562590"
    ),
    row!(
        L690,
        SelectionBranch,
        r#"VAE.__init__.selection@L690:"decoder.conv_in.conv.weight" in sd and "decoder.mid_block.resnets.0.norm1.norm_layer.weight" in sd"#,
        74,
        "cec0b2bd04efe784de929181caac16ad0f141ac74ec828e26865321a9b27c71b"
    ),
    row!(
        L701,
        SelectionBranch,
        r#"VAE.__init__.selection@L701:"decoder.conv_in.conv.weight" in sd"#,
        75,
        "452673f09e0172bf0c667ec5f8513efc17abcb7ceb2ed7f23890956dbedc7df8"
    ),
    row!(
        L717,
        SelectionBranch,
        r#"VAE.__init__.selection@L717:"decoder.unpatcher3d.wavelets" in sd"#,
        76,
        "2b69b27a2b994785428fcb80b0c7fcc4d4cfd7db928bdf439ac2274b90ebbd85"
    ),
    row!(
        L730,
        SelectionBranch,
        r#"VAE.__init__.selection@L730:"decoder.middle.0.residual.0.gamma" in sd"#,
        77,
        "c3089c7a3ba03b8c8996c364aacfb445348f18ab749932e5d3c22909b37bb8f7"
    ),
    row!(
        L731,
        SelectionBranch,
        r#"VAE.__init__.selection@L731:"decoder.upsamples.0.upsamples.0.residual.2.weight" in sd"#,
        78,
        "437f41191989c7a9e5874437afa4a33ca32e1d2d50af028bc40616ad2177424b"
    ),
    row!(
        L762,
        SelectionBranch,
        r#"VAE.__init__.selection@L762:"geo_decoder.cross_attn_decoder.ln_1.bias" in sd"#,
        79,
        "43c03e714eb722443d11733961ad9be07e3cd4ef290c2e264cfe8a40a5534d41"
    ),
    row!(
        L784,
        SelectionBranch,
        r#"VAE.__init__.selection@L784:"vocoder.backbone.channel_layers.0.0.bias" in sd"#,
        80,
        "b5c02f4513d445c62fe170d36cf11c8b4320c6e98c1e87dd9a5e7670b6cc8483"
    ),
    row!(
        L799,
        SelectionBranch,
        r#"VAE.__init__.selection@L799:"pixel_space_vae" in sd"#,
        81,
        "6dab3ef25a9ce66a85ba664de36cc28e828adc2b6a3391b109759db191e11333"
    ),
    row!(
        L809,
        SelectionBranch,
        r#"VAE.__init__.selection@L809:"vocoder.activation_post.downsample.lowpass.filter" in sd"#,
        82,
        "d910ac1e204a00f6a0a1ef35706bac9e7dc1436294a08711d3a5aa3070bf5ab0"
    ),
    row!(
        L828,
        SelectionBranch,
        r#"VAE.__init__.selection@L828:"decoder.22.bias" in sd"#,
        83,
        "1fd21ffde691eea7d834170d29d37a3b7bf03886405211da5b6a3528d5c89a78"
    ),
    row!(
        L835,
        SelectionBranch,
        "VAE.__init__.selection@L835:self.latent_channels in [48, 128]",
        84,
        "137221035de6d9e13ce0c3a40b48093d3e0ae5c04fdb6d391681884d22528a23"
    ),
    row!(
        L840,
        SelectionBranch,
        r#"VAE.__init__.selection@L840:self.latent_channels == 32 and sd["decoder.22.bias"].shape[0] == 12"#,
        85,
        "9deedd505128e58ef18b8c45ef26003722ecbfe9dacc106b295f9186e4a96614"
    ),
    row!(
        L856,
        SelectionBranch,
        r#"VAE.__init__.selection@L856:"vocoder.resblocks.0.convs1.0.weight" in sd or "vocoder.vocoder.resblocks.0.convs1.0.weight" in sd"#,
        86,
        "660caf2731a1a18157a6d7f558b5f54a0c1149553fb53eb3e6e772897320c8f9"
    ),
    row!(
        L874,
        SelectionBranch,
        r#"VAE.__init__.selection@L874:"decoder.layers.3.transformers.0.pre_norm.alpha" in sd"#,
        87,
        "488f5c08fbf12f5545fb47a06e8ad8fbc4395cae6ae4412db8ca5609aebc4b0c"
    ),
    row!(
        L902,
        SelectionBranch,
        r#"VAE.__init__.selection@L902:"gs.base_offset_scale" in sd and "octree.out_proj.weight" in sd"#,
        88,
        "3e31f0148dbec928096fc5a6d1491b4a906dc034943b1ef9260c088b86e59933"
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VaeBoundaryDomain {
    Image,
    Video,
    Audio,
    Structured,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VaeLoaderConfiguration {
    Automatic,
    DiffusersPreconverted {
        key_mapping: Vec<(String, String)>,
        conv3d: bool,
        inner: Box<VaeLoaderConfiguration>,
    },
    Taesd {
        latent_channels: u64,
        metadata_override: bool,
    },
    DefaultKl {
        x4: bool,
        legacy_prefix_rewrite: bool,
        batch_norm_latent: bool,
        asymmetric_decoder_channels: Option<u64>,
        embed_dim: Option<u64>,
    },
    LtxVideo {
        configuration_sha256: Option<String>,
        configuration_json: Option<String>,
    },
    LtxAudio {
        autoencoder_sha256: String,
        autoencoder_json: String,
        vocoder_sha256: String,
        vocoder_json: String,
        latent_channels: u64,
        input_sample_rate: u32,
        output_sample_rate: u32,
    },
    ExplicitAutoencoderKl {
        params_sha256: String,
        params_json: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExplicitAutoencoderKlSide {
    pub(crate) base_channels: u64,
    pub(crate) channel_multipliers: Vec<u64>,
    pub(crate) residual_blocks: u64,
    pub(crate) boundary_channels: u64,
    pub(crate) latent_channels: u64,
    pub(crate) attention_levels: Vec<usize>,
    pub(crate) resample_with_convolution: bool,
    pub(crate) tanh_output: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExplicitAutoencoderKlTopology {
    pub(crate) encoder: ExplicitAutoencoderKlSide,
    pub(crate) decoder: ExplicitAutoencoderKlSide,
    pub(crate) embed_dim: u64,
    pub(crate) batch_norm_latent: bool,
}

impl ExplicitAutoencoderKlTopology {
    pub(crate) fn parse(params_json: &str) -> Result<Self, VaeArchitectureError> {
        let params = serde_json::from_str::<serde_json::Value>(params_json)
            .map_err(|error| VaeArchitectureError::MalformedConfiguration(error.to_string()))?;
        let params = params.as_object().ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(
                "explicit AutoencoderKL params must be an object".to_owned(),
            )
        })?;
        let encoder = params
            .get("ddconfig")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(
                    "explicit AutoencoderKL params require an object-valued ddconfig".to_owned(),
                )
            })?;
        let decoder = params
            .get("decoder_ddconfig")
            .map(|value| {
                value.as_object().ok_or_else(|| {
                    VaeArchitectureError::MalformedConfiguration(
                        "explicit AutoencoderKL decoder_ddconfig must be an object".to_owned(),
                    )
                })
            })
            .transpose()?
            .unwrap_or(encoder);
        let double_latent = explicit_required_bool(encoder, "double_z")?;
        if !double_latent {
            return Err(VaeArchitectureError::MalformedConfiguration(
                "explicit AutoencoderKL requires ddconfig.double_z=true for bidirectional mode/decode channel parity"
                    .to_owned(),
            ));
        }
        reject_explicit_video_configuration(encoder, "ddconfig")?;
        reject_explicit_video_configuration(decoder, "decoder_ddconfig")?;
        for (configuration, label) in [(encoder, "ddconfig"), (decoder, "decoder_ddconfig")] {
            if configuration
                .get("use_linear_attn")
                .is_some_and(|value| value.as_bool() != Some(false))
                || configuration
                    .get("attn_type")
                    .is_some_and(|value| value.as_str() != Some("vanilla"))
            {
                return Err(VaeArchitectureError::MalformedConfiguration(format!(
                    "explicit image AutoencoderKL {label} supports only use_linear_attn=false and attn_type=vanilla"
                )));
            }
        }
        let embed_dim = explicit_required_u64(params, "embed_dim", "params")?;
        let batch_norm_latent = encoder
            .get("batch_norm_latent")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    VaeArchitectureError::MalformedConfiguration(
                        "explicit AutoencoderKL ddconfig.batch_norm_latent must be boolean"
                            .to_owned(),
                    )
                })
            })
            .transpose()?
            .unwrap_or(false);
        let encoder = parse_explicit_autoencoder_kl_side(encoder, "ddconfig", true)?;
        let decoder = parse_explicit_autoencoder_kl_side(decoder, "decoder_ddconfig", false)?;
        if batch_norm_latent && encoder.latent_channels != embed_dim {
            return Err(VaeArchitectureError::MalformedConfiguration(
                "explicit AutoencoderKL batch normalization requires embed_dim to equal ddconfig.z_channels"
                    .to_owned(),
            ));
        }
        Ok(Self {
            encoder,
            decoder,
            embed_dim,
            batch_norm_latent,
        })
    }

    pub(crate) fn encode_ratio(&self) -> Result<u64, VaeArchitectureError> {
        explicit_side_ratio(&self.encoder, self.batch_norm_latent)
    }

    pub(crate) fn decode_ratio(&self) -> Result<u64, VaeArchitectureError> {
        explicit_side_ratio(&self.decoder, self.batch_norm_latent)
    }

    pub(crate) fn public_latent_channels(&self) -> Result<u64, VaeArchitectureError> {
        if self.batch_norm_latent {
            self.embed_dim.checked_mul(4).ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(
                    "explicit AutoencoderKL public latent channels overflow".to_owned(),
                )
            })
        } else {
            Ok(self.embed_dim)
        }
    }
}

fn parse_explicit_autoencoder_kl_side(
    configuration: &serde_json::Map<String, serde_json::Value>,
    label: &'static str,
    encoder: bool,
) -> Result<ExplicitAutoencoderKlSide, VaeArchitectureError> {
    let base_channels = explicit_required_u64(configuration, "ch", label)?;
    let channel_multipliers = explicit_required_u64_array(configuration, "ch_mult", label)?;
    let residual_blocks = explicit_required_u64(configuration, "num_res_blocks", label)?;
    let boundary_channels = explicit_required_u64(
        configuration,
        if encoder { "in_channels" } else { "out_ch" },
        label,
    )?;
    let latent_channels = explicit_required_u64(configuration, "z_channels", label)?;
    let resolution = explicit_required_u64(configuration, "resolution", label)?;
    let attention_resolutions =
        explicit_required_u64_array(configuration, "attn_resolutions", label)?;
    let resample_with_convolution =
        explicit_optional_bool(configuration, "resamp_with_conv", label, true)?;
    let tanh_output = if encoder {
        false
    } else {
        explicit_optional_bool(configuration, "tanh_out", label, false)?
    };
    if channel_multipliers.is_empty() {
        return Err(VaeArchitectureError::MalformedConfiguration(format!(
            "explicit AutoencoderKL {label}.ch_mult must not be empty"
        )));
    }
    if residual_blocks == 0 {
        return Err(VaeArchitectureError::MalformedConfiguration(format!(
            "explicit AutoencoderKL {label}.num_res_blocks must be positive"
        )));
    }
    let mut current_resolution = resolution;
    let mut attention_levels = Vec::new();
    for (level, multiplier) in channel_multipliers.iter().copied().enumerate() {
        let channels = base_channels.checked_mul(multiplier).ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(format!(
                "explicit AutoencoderKL {label} channel count overflows"
            ))
        })?;
        if !channels.is_multiple_of(32) {
            return Err(VaeArchitectureError::MalformedConfiguration(format!(
                "explicit AutoencoderKL {label} normalized channel count {channels} is not divisible by 32"
            )));
        }
        if attention_resolutions.contains(&current_resolution) {
            attention_levels.push(level);
        }
        if level + 1 < channel_multipliers.len() {
            current_resolution = current_resolution.checked_div(2).ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(format!(
                    "explicit AutoencoderKL {label} resolution underflow"
                ))
            })?;
            if current_resolution == 0 {
                return Err(VaeArchitectureError::MalformedConfiguration(format!(
                    "explicit AutoencoderKL {label} resolution is too small for ch_mult"
                )));
            }
        }
    }
    Ok(ExplicitAutoencoderKlSide {
        base_channels,
        channel_multipliers,
        residual_blocks,
        boundary_channels,
        latent_channels,
        attention_levels,
        resample_with_convolution,
        tanh_output,
    })
}

fn explicit_side_ratio(
    side: &ExplicitAutoencoderKlSide,
    batch_norm_latent: bool,
) -> Result<u64, VaeArchitectureError> {
    let shifts = u32::try_from(side.channel_multipliers.len().saturating_sub(1)).map_err(|_| {
        VaeArchitectureError::MalformedConfiguration(
            "explicit AutoencoderKL spatial ratio overflows".to_owned(),
        )
    })?;
    let ratio = 1_u64.checked_shl(shifts).ok_or_else(|| {
        VaeArchitectureError::MalformedConfiguration(
            "explicit AutoencoderKL spatial ratio overflows".to_owned(),
        )
    })?;
    if batch_norm_latent {
        ratio.checked_mul(2).ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(
                "explicit AutoencoderKL batch-normalized spatial ratio overflows".to_owned(),
            )
        })
    } else {
        Ok(ratio)
    }
}

fn explicit_required_u64(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    label: &'static str,
) -> Result<u64, VaeArchitectureError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(format!(
                "explicit AutoencoderKL {label}.{field} must be a positive unsigned integer"
            ))
        })
}

fn explicit_required_u64_array(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    label: &'static str,
) -> Result<Vec<u64>, VaeArchitectureError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(format!(
                "explicit AutoencoderKL {label}.{field} must be an array"
            ))
        })?
        .iter()
        .map(|value| {
            value.as_u64().filter(|value| *value > 0).ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(format!(
                    "explicit AutoencoderKL {label}.{field} values must be positive unsigned integers"
                ))
            })
        })
        .collect()
}

fn explicit_required_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
) -> Result<bool, VaeArchitectureError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            VaeArchitectureError::MalformedConfiguration(format!(
                "explicit AutoencoderKL ddconfig.{field} must be boolean"
            ))
        })
}

fn explicit_optional_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    label: &'static str,
    default: bool,
) -> Result<bool, VaeArchitectureError> {
    object
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(format!(
                    "explicit AutoencoderKL {label}.{field} must be boolean"
                ))
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn reject_explicit_video_configuration(
    object: &serde_json::Map<String, serde_json::Value>,
    label: &'static str,
) -> Result<(), VaeArchitectureError> {
    if object
        .get("conv3d")
        .is_some_and(|value| value.as_bool() != Some(false))
        || object
            .get("time_compress")
            .is_some_and(|value| !value.is_null())
    {
        return Err(VaeArchitectureError::MalformedConfiguration(format!(
            "explicit image AutoencoderKL {label} cannot enable conv3d or time_compress"
        )));
    }
    Ok(())
}

impl VaeLoaderConfiguration {
    pub fn owned_resident_bytes(&self) -> Option<u64> {
        match self {
            Self::Automatic | Self::Taesd { .. } | Self::DefaultKl { .. } => Some(0),
            Self::DiffusersPreconverted {
                key_mapping, inner, ..
            } => {
                let inline = key_mapping
                    .capacity()
                    .checked_mul(std::mem::size_of::<(String, String)>())?;
                let mappings = key_mapping.iter().try_fold(
                    u64::try_from(inline).ok()?,
                    |total, (source, target)| {
                        let strings = source.capacity().checked_add(target.capacity())?;
                        total.checked_add(u64::try_from(strings).ok()?)
                    },
                )?;
                mappings
                    .checked_add(u64::try_from(std::mem::size_of::<Self>()).ok()?)?
                    .checked_add(inner.owned_resident_bytes()?)
            }
            Self::LtxVideo {
                configuration_sha256,
                configuration_json,
            } => owned_optional_string_bytes(configuration_sha256).and_then(|bytes| {
                bytes.checked_add(owned_optional_string_bytes(configuration_json)?)
            }),
            Self::LtxAudio {
                autoencoder_sha256,
                autoencoder_json,
                vocoder_sha256,
                vocoder_json,
                ..
            } => owned_string_bytes([
                autoencoder_sha256,
                autoencoder_json,
                vocoder_sha256,
                vocoder_json,
            ]),
            Self::ExplicitAutoencoderKl {
                params_sha256,
                params_json,
            } => owned_string_bytes([params_sha256, params_json]),
        }
    }

    pub fn digest(&self) -> Result<String, VaeArchitectureError> {
        let encoded = serde_json::to_vec(self)
            .map_err(|error| VaeArchitectureError::Registry(error.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }

    pub(crate) fn validate(&self) -> Result<(), VaeArchitectureError> {
        match self {
            Self::Automatic => Ok(()),
            Self::DiffusersPreconverted {
                key_mapping, inner, ..
            } => {
                if key_mapping.is_empty()
                    || key_mapping.len() > 1_000_000
                    || key_mapping.iter().any(|(source, target)| {
                        source.is_empty()
                            || target.is_empty()
                            || source.len() > 4_096
                            || target.len() > 4_096
                    })
                {
                    return Err(VaeArchitectureError::MalformedConfiguration(
                        "Diffusers key mapping is empty or exceeds bounded fields".to_owned(),
                    ));
                }
                let source_count = key_mapping
                    .iter()
                    .map(|(source, _)| source)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                let target_count = key_mapping
                    .iter()
                    .map(|(_, target)| target)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                if source_count != key_mapping.len() || target_count != key_mapping.len() {
                    return Err(VaeArchitectureError::MalformedConfiguration(
                        "Diffusers key mapping contains duplicate endpoints".to_owned(),
                    ));
                }
                inner.validate()
            }
            Self::Taesd {
                latent_channels, ..
            } if *latent_channels == 0 => Err(VaeArchitectureError::MalformedConfiguration(
                "TAESD latent channels must be positive".to_owned(),
            )),
            Self::LtxVideo {
                configuration_sha256,
                configuration_json,
            } => validate_retained_configuration(configuration_sha256, configuration_json),
            Self::LtxAudio {
                autoencoder_sha256,
                autoencoder_json,
                vocoder_sha256,
                vocoder_json,
                latent_channels,
                input_sample_rate,
                output_sample_rate,
            } => {
                validate_retained_configuration(
                    &Some(autoencoder_sha256.clone()),
                    &Some(autoencoder_json.clone()),
                )?;
                validate_retained_configuration(
                    &Some(vocoder_sha256.clone()),
                    &Some(vocoder_json.clone()),
                )?;
                if *latent_channels == 0 || *input_sample_rate == 0 || *output_sample_rate == 0 {
                    return Err(VaeArchitectureError::MalformedConfiguration(
                        "LTX Audio channels and rates must be positive".to_owned(),
                    ));
                }
                Ok(())
            }
            Self::ExplicitAutoencoderKl {
                params_sha256,
                params_json,
            } => {
                validate_retained_configuration(
                    &Some(params_sha256.clone()),
                    &Some(params_json.clone()),
                )?;
                ExplicitAutoencoderKlTopology::parse(params_json)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn validate_for_profile(
        &self,
        profile: &VaeKernelProfile,
    ) -> Result<(), VaeArchitectureError> {
        self.validate()?;
        let compatible = match self {
            Self::Automatic => !matches!(
                profile,
                VaeKernelProfile::TaesdV1
                    | VaeKernelProfile::AutoencoderKlV1
                    | VaeKernelProfile::AutoencoderKlX4V1
                    | VaeKernelProfile::AutoencoderKlBatchNormV1
                    | VaeKernelProfile::ExplicitAutoencoderKlV1
                    | VaeKernelProfile::AutoencodingEngineV1
                    | VaeKernelProfile::AutoencodingEngineX4V1
                    | VaeKernelProfile::AutoencodingEngineBatchNormV1
                    | VaeKernelProfile::LtxVideoV0 { .. }
                    | VaeKernelProfile::LtxVideoV1 { .. }
                    | VaeKernelProfile::LtxVideoV2 { .. }
                    | VaeKernelProfile::LtxAudioV1
            ),
            Self::DiffusersPreconverted { inner, .. } => {
                inner.validate_for_profile(profile).is_ok()
            }
            Self::Taesd { .. } => matches!(profile, VaeKernelProfile::TaesdV1),
            Self::DefaultKl {
                x4,
                batch_norm_latent,
                ..
            } => match profile {
                VaeKernelProfile::AutoencoderKlV1 | VaeKernelProfile::AutoencodingEngineV1 => {
                    !x4 && !batch_norm_latent
                }
                VaeKernelProfile::AutoencoderKlX4V1 | VaeKernelProfile::AutoencodingEngineX4V1 => {
                    *x4 && !batch_norm_latent
                }
                VaeKernelProfile::AutoencoderKlBatchNormV1
                | VaeKernelProfile::AutoencodingEngineBatchNormV1 => *batch_norm_latent,
                _ => false,
            },
            Self::LtxVideo {
                configuration_sha256,
                ..
            } => match profile {
                VaeKernelProfile::LtxVideoV0 {
                    configuration_sha256: profile_digest,
                }
                | VaeKernelProfile::LtxVideoV1 {
                    configuration_sha256: profile_digest,
                }
                | VaeKernelProfile::LtxVideoV2 {
                    configuration_sha256: profile_digest,
                } => profile_digest == configuration_sha256,
                _ => false,
            },
            Self::LtxAudio { .. } => matches!(profile, VaeKernelProfile::LtxAudioV1),
            Self::ExplicitAutoencoderKl { .. } => {
                matches!(profile, VaeKernelProfile::ExplicitAutoencoderKlV1)
            }
        };
        if compatible {
            Ok(())
        } else {
            Err(VaeArchitectureError::MalformedConfiguration(format!(
                "loader configuration is incompatible with VAE profile {profile:?}"
            )))
        }
    }
}

fn owned_optional_string_bytes(value: &Option<String>) -> Option<u64> {
    u64::try_from(value.as_ref().map_or(0, String::capacity)).ok()
}

fn owned_string_bytes<'a>(values: impl IntoIterator<Item = &'a String>) -> Option<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(u64::try_from(value.capacity()).ok()?)
    })
}

fn validate_retained_configuration(
    digest: &Option<String>,
    json: &Option<String>,
) -> Result<(), VaeArchitectureError> {
    match (digest, json) {
        (None, None) => Ok(()),
        (Some(digest), Some(json)) if json.len() <= MAX_VAE_CONFIGURATION_BYTES => {
            validate_sha256(digest)?;
            let value: serde_json::Value = serde_json::from_str(json)
                .map_err(|error| VaeArchitectureError::MalformedConfiguration(error.to_string()))?;
            let canonical = canonical_json(&value)?;
            if canonical != *json || format!("{:x}", Sha256::digest(json.as_bytes())) != *digest {
                return Err(VaeArchitectureError::MalformedConfiguration(
                    "retained configuration is not canonical or digest-bound".to_owned(),
                ));
            }
            Ok(())
        }
        _ => Err(VaeArchitectureError::MalformedConfiguration(
            "retained configuration digest/data pair is incomplete or oversized".to_owned(),
        )),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaeArchitectureSelection {
    architecture: VaeArchitectureIdentity,
    profile: VaeKernelProfile,
    trace: Vec<&'static str>,
    latent_channels: Option<u64>,
    target_latent_channels: Option<u64>,
    latent_dimensions: u8,
    boundary: VaeBoundaryDomain,
    canonical_compatibility: VaeCanonicalCompatibility,
    loader_configuration: VaeLoaderConfiguration,
    supported_dtypes: &'static [DType],
    supported_devices: &'static [DeviceKind],
}

impl VaeArchitectureSelection {
    pub fn architecture(&self) -> &VaeArchitectureIdentity {
        &self.architecture
    }

    pub fn profile(&self) -> &VaeKernelProfile {
        &self.profile
    }

    pub fn trace(&self) -> &[&'static str] {
        &self.trace
    }

    pub fn latent_channels(&self) -> Option<u64> {
        self.latent_channels
    }

    pub fn target_latent_channels(&self) -> Option<u64> {
        self.target_latent_channels
    }

    pub fn latent_dimensions(&self) -> u8 {
        self.latent_dimensions
    }

    pub fn boundary(&self) -> VaeBoundaryDomain {
        self.boundary
    }

    pub fn canonical_compatibility(&self) -> VaeCanonicalCompatibility {
        self.canonical_compatibility
    }

    pub fn loader_configuration(&self) -> &VaeLoaderConfiguration {
        &self.loader_configuration
    }

    pub fn supported_dtypes(&self) -> &'static [DType] {
        self.supported_dtypes
    }

    pub fn supported_devices(&self) -> &'static [DeviceKind] {
        self.supported_devices
    }

    pub fn ensure_native_builder_available(&self) -> Result<(), VaeArchitectureError> {
        self.loader_configuration
            .validate_for_profile(&self.profile)?;
        if matches!(
            self.profile,
            VaeKernelProfile::TemporalAutoencodingEngineV1
                | VaeKernelProfile::TaesdV1
                | VaeKernelProfile::StableCascadeStageAV1
                | VaeKernelProfile::StableCascadeStageCEncoderV1
                | VaeKernelProfile::StableCascadeStageCPreviewerV1
                | VaeKernelProfile::StableCascadeStageCCombinedV1
                | VaeKernelProfile::HunyuanImageV1
                | VaeKernelProfile::AutoencoderKlV1
                | VaeKernelProfile::AutoencoderKlX4V1
                | VaeKernelProfile::AutoencoderKlBatchNormV1
                | VaeKernelProfile::ExplicitAutoencoderKlV1
                | VaeKernelProfile::AutoencodingEngineV1
                | VaeKernelProfile::AutoencodingEngineX4V1
                | VaeKernelProfile::AutoencodingEngineBatchNormV1
                | VaeKernelProfile::PixelSpaceV1
                | VaeKernelProfile::TaeHvWan22V1
                | VaeKernelProfile::TaeHvLtx2V1
                | VaeKernelProfile::LightTaeHv15V1
                | VaeKernelProfile::TaeHvHunyuanV1
                | VaeKernelProfile::LightTaeWan21V1
                | VaeKernelProfile::HunyuanImageRefinerV1
                | VaeKernelProfile::HunyuanVideoRefinerV1
                | VaeKernelProfile::Causal3dV1
                | VaeKernelProfile::CogVideoXV1
                | VaeKernelProfile::CosmosV1
                | VaeKernelProfile::MochiV1
                | VaeKernelProfile::LtxVideoV0 { .. }
                | VaeKernelProfile::LtxVideoV1 { .. }
                | VaeKernelProfile::LtxVideoV2 { .. }
                | VaeKernelProfile::Wan21V1
                | VaeKernelProfile::Wan22V1
                | VaeKernelProfile::AudioOobleck44KhzV1
                | VaeKernelProfile::AudioOobleck48KhzV1
                | VaeKernelProfile::MusicDcaeV1
                | VaeKernelProfile::MmAudio16KhzV1
                | VaeKernelProfile::LtxAudioV1
                | VaeKernelProfile::StableAudio3DeepV1
                | VaeKernelProfile::StableAudio3ShallowV1
                | VaeKernelProfile::HunyuanShapeV1
                | VaeKernelProfile::TripoSplatV1
        ) {
            Ok(())
        } else {
            Err(VaeArchitectureError::ArchitectureUnavailable {
                architecture: self.architecture.as_str().to_owned(),
            })
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaeExecutionTarget {
    family: ModelFamilyIdentity,
    latent_format: LatentFormatIdentity,
    dtype: DType,
    device: DeviceId,
}

impl VaeExecutionTarget {
    pub fn new(
        family: ModelFamilyIdentity,
        latent_format: LatentFormatIdentity,
        dtype: DType,
        device: DeviceId,
    ) -> Self {
        Self {
            family,
            latent_format,
            dtype,
            device,
        }
    }

    pub fn family(&self) -> &ModelFamilyIdentity {
        &self.family
    }

    pub fn latent_format(&self) -> &LatentFormatIdentity {
        &self.latent_format
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }
}

#[derive(Clone, Debug, Default)]
pub struct VaeArchitectureRegistry;

impl VaeArchitectureRegistry {
    pub fn checked() -> Result<Self, VaeArchitectureError> {
        if VAE_SELECTOR_CATALOG_ROWS.len() != VAE_SELECTOR_ROW_COUNT {
            return Err(VaeArchitectureError::Registry(
                "catalog row count does not match the pinned closure".to_owned(),
            ));
        }
        if VAE_SELECTOR_CATALOG_ROWS
            .windows(2)
            .any(|rows| rows[0].source_ordinal >= rows[1].source_ordinal)
        {
            return Err(VaeArchitectureError::Registry(
                "catalog rows are not in strict source order".to_owned(),
            ));
        }
        for row in VAE_SELECTOR_CATALOG_ROWS {
            validate_sha256(row.symbol_sha256)?;
        }
        validate_sha256(VAE_SELECTOR_SOURCE_SHA256)?;
        Ok(Self)
    }

    pub fn canonical_targets()
    -> Result<(ModelFamilyRegistry, LatentFormatRegistry), VaeArchitectureError> {
        Ok((
            ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)
                .map_err(|error| VaeArchitectureError::FamilyRegistry(error.to_string()))?,
            LatentFormatRegistry::checked(GENERATED_LATENT_FORMATS)
                .map_err(|error| VaeArchitectureError::LatentRegistry(error.to_string()))?,
        ))
    }

    pub fn select(
        &self,
        probe: &ModelProbe,
        cancellation: &CancellationToken,
    ) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
        cancellation.check()?;
        let (normalized_probe, preconversion) = normalize_diffusers_probe(probe, cancellation)?;
        let probe = normalized_probe.as_ref().unwrap_or(probe);
        let signals = primary_signals(probe);
        if signals.is_empty() {
            return Err(VaeArchitectureError::NoMatch {
                unbound_row: VAE_UNBOUND_ROW_ID,
            });
        }
        if signals.len() > 1 {
            return Err(VaeArchitectureError::Ambiguous {
                rows: signals
                    .iter()
                    .map(|signal| signal.row_for_probe(probe))
                    .collect(),
            });
        }
        cancellation.check()?;
        let mut selection = select_signal(signals[0], probe)?;
        if let Some(preconversion) = preconversion {
            selection.trace.insert(0, VAE_DIFFUSERS_ROW_ID);
            selection.loader_configuration = VaeLoaderConfiguration::DiffusersPreconverted {
                key_mapping: preconversion.key_mapping,
                conv3d: preconversion.conv3d,
                inner: Box::new(selection.loader_configuration),
            };
        }
        selection.trace.insert(0, VAE_AUTOMATIC_ROW_ID);
        cancellation.check()?;
        Ok(selection)
    }

    pub fn select_explicit(
        &self,
        probe: &ModelProbe,
        config: &str,
        cancellation: &CancellationToken,
    ) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
        cancellation.check()?;
        if config.len() > MAX_VAE_CONFIGURATION_BYTES {
            return Err(VaeArchitectureError::ConfigurationLimit {
                actual: config.len(),
                maximum: MAX_VAE_CONFIGURATION_BYTES,
            });
        }
        let root: serde_json::Value = serde_json::from_str(config)
            .map_err(|error| VaeArchitectureError::MalformedConfiguration(error.to_string()))?;
        let params = root
            .as_object()
            .and_then(|object| object.get("params"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                VaeArchitectureError::MalformedConfiguration(
                    "expected an object-valued params field".to_owned(),
                )
            })?;
        let (_, preconversion) = normalize_diffusers_probe(probe, cancellation)?;
        let params_json = canonical_json(&serde_json::Value::Object(params.clone()))?;
        let mut selection = selected(
            "sim.ldm.models.autoencoder.AutoencoderKL.v1",
            VaeKernelProfile::ExplicitAutoencoderKlV1,
            Vec::new(),
            None,
            2,
            VaeBoundaryDomain::Image,
        )?;
        selection.loader_configuration = VaeLoaderConfiguration::ExplicitAutoencoderKl {
            params_sha256: format!("{:x}", Sha256::digest(params_json.as_bytes())),
            params_json,
        };
        selection
            .loader_configuration
            .validate_for_profile(&selection.profile)?;
        if let Some(preconversion) = preconversion {
            selection.trace.push(VAE_DIFFUSERS_ROW_ID);
            selection.loader_configuration = VaeLoaderConfiguration::DiffusersPreconverted {
                key_mapping: preconversion.key_mapping,
                conv3d: preconversion.conv3d,
                inner: Box::new(selection.loader_configuration),
            };
        }
        cancellation.check()?;
        Ok(selection)
    }

    pub fn select_for_target(
        &self,
        probe: &ModelProbe,
        target: &VaeExecutionTarget,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        cancellation: &CancellationToken,
    ) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
        let selection = self.select(probe, cancellation)?;
        self.validate_target(
            &selection,
            target,
            family_registry,
            latent_registry,
            cancellation,
        )?;
        Ok(selection)
    }

    pub fn select_loaded(
        &self,
        store: &ModelStore,
        loaded_model: &Arc<LoadedModel>,
        target: &VaeExecutionTarget,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        cancellation: &CancellationToken,
    ) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
        cancellation.check()?;
        let probe = store.family_probe(loaded_model, cancellation)?;
        self.select_for_target(
            &probe,
            target,
            family_registry,
            latent_registry,
            cancellation,
        )
    }

    pub fn intended_target(
        &self,
        selection: &VaeArchitectureSelection,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        cancellation: &CancellationToken,
    ) -> Result<VaeExecutionTarget, VaeArchitectureError> {
        cancellation.check()?;
        let allowed = match selection.canonical_compatibility {
            VaeCanonicalCompatibility::Exact(allowed) => allowed,
            VaeCanonicalCompatibility::Unavailable(reason) => {
                return Err(VaeArchitectureError::CanonicalTargetUnavailable {
                    architecture: selection.architecture.as_str().to_owned(),
                    reason,
                });
            }
        };
        for family in family_registry.definitions_in_source_order() {
            cancellation.check()?;
            if !allowed.contains(&family.latent_identifier)
                || !family.supported_dtypes.contains(&DType::F32)
                || !family.supported_devices.contains(&DeviceKind::Cpu)
                || !selection.supported_dtypes.contains(&DType::F32)
                || !selection.supported_devices.contains(&DeviceKind::Cpu)
            {
                continue;
            }
            let latent_identity =
                LatentFormatIdentity::new(family.latent_feature_id, family.latent_identifier)
                    .map_err(|error| VaeArchitectureError::LatentRegistry(error.to_string()))?;
            let Some(latent) = latent_registry.get(&latent_identity) else {
                continue;
            };
            if selection.latent_dimensions != latent.dimensions
                || selection
                    .target_latent_channels
                    .is_some_and(|channels| channels != latent.channels)
                || !selection_configuration_matches_latent(selection, latent)?
            {
                continue;
            }
            return Ok(VaeExecutionTarget::new(
                ModelFamilyIdentity::new(
                    family.feature_id,
                    family.identifier,
                    family.architecture_version,
                )
                .map_err(|error| VaeArchitectureError::FamilyRegistry(error.to_string()))?,
                latent_identity,
                DType::F32,
                DeviceId::CPU,
            ));
        }
        Err(VaeArchitectureError::CanonicalFamilyUnavailable {
            architecture: selection.architecture.as_str().to_owned(),
            allowed: allowed.to_vec(),
        })
    }

    pub fn validate_target(
        &self,
        selection: &VaeArchitectureSelection,
        target: &VaeExecutionTarget,
        family_registry: &ModelFamilyRegistry,
        latent_registry: &LatentFormatRegistry,
        cancellation: &CancellationToken,
    ) -> Result<(), VaeArchitectureError> {
        cancellation.check()?;
        let family = family_registry.definition(&target.family).ok_or_else(|| {
            VaeArchitectureError::UnknownModelFamily(target.family.identifier().to_owned())
        })?;
        let latent = latent_registry.get(&target.latent_format).ok_or_else(|| {
            VaeArchitectureError::UnknownLatentFormat(target.latent_format.identifier().to_owned())
        })?;
        if family.latent_feature_id != target.latent_format.feature_id()
            || family.latent_identifier != target.latent_format.identifier()
        {
            return Err(VaeArchitectureError::FamilyLatentMismatch {
                family: family.identifier.to_owned(),
                expected: format!("{}:{}", family.latent_feature_id, family.latent_identifier),
                actual: format!(
                    "{}:{}",
                    target.latent_format.feature_id(),
                    target.latent_format.identifier()
                ),
            });
        }
        let allowed = match selection.canonical_compatibility {
            VaeCanonicalCompatibility::Exact(allowed) => allowed,
            VaeCanonicalCompatibility::Unavailable(reason) => {
                return Err(VaeArchitectureError::CanonicalTargetUnavailable {
                    architecture: selection.architecture.as_str().to_owned(),
                    reason,
                });
            }
        };
        if selection.latent_dimensions != latent.dimensions
            || selection
                .target_latent_channels
                .is_some_and(|channels| channels != latent.channels)
            || !selection_configuration_matches_latent(selection, latent)?
            || !allowed.contains(&latent.identifier)
        {
            return Err(VaeArchitectureError::ProfileLatentMismatch {
                architecture: selection.architecture.as_str().to_owned(),
                latent: latent.identifier.to_owned(),
            });
        }
        if !family.supported_dtypes.contains(&target.dtype) {
            return Err(VaeArchitectureError::UnsupportedTargetDType {
                family: family.identifier.to_owned(),
                dtype: target.dtype,
            });
        }
        if !selection.supported_dtypes.contains(&target.dtype) {
            return Err(VaeArchitectureError::UnsupportedProfileDType {
                architecture: selection.architecture.as_str().to_owned(),
                dtype: target.dtype,
            });
        }
        let device_kind = target.device.kind();
        if !family.supported_devices.contains(&device_kind) {
            return Err(VaeArchitectureError::UnsupportedTargetDevice {
                family: family.identifier.to_owned(),
                device: device_kind,
            });
        }
        if !selection.supported_devices.contains(&device_kind) {
            return Err(VaeArchitectureError::UnsupportedProfileDevice {
                architecture: selection.architecture.as_str().to_owned(),
                device: device_kind,
            });
        }
        cancellation.check()?;
        Ok(())
    }
}

pub(crate) fn checked_recipe_configuration_is_explicit(
    configuration: &str,
) -> Result<bool, VaeArchitectureError> {
    if configuration.len() > MAX_VAE_CONFIGURATION_BYTES {
        return Err(VaeArchitectureError::ConfigurationLimit {
            actual: configuration.len(),
            maximum: MAX_VAE_CONFIGURATION_BYTES,
        });
    }
    let document: serde_json::Value = serde_json::from_str(configuration)
        .map_err(|error| VaeArchitectureError::MalformedConfiguration(error.to_string()))?;
    let object = document.as_object().ok_or_else(|| {
        VaeArchitectureError::MalformedConfiguration(
            "recipe configuration must be a JSON object".to_owned(),
        )
    })?;
    Ok(object
        .get("params")
        .is_some_and(serde_json::Value::is_object))
}

fn selection_configuration_matches_latent(
    selection: &VaeArchitectureSelection,
    latent: &crate::LatentFormatDefinition,
) -> Result<bool, VaeArchitectureError> {
    let configuration = match selection.loader_configuration() {
        VaeLoaderConfiguration::DiffusersPreconverted { inner, .. } => inner.as_ref(),
        configuration => configuration,
    };
    match configuration {
        VaeLoaderConfiguration::DefaultKl {
            x4,
            batch_norm_latent,
            embed_dim,
            ..
        } => {
            let base_ratio = if *x4 { 4_u64 } else { 8_u64 };
            let spatial_ratio = if *batch_norm_latent {
                base_ratio.checked_mul(2).ok_or_else(|| {
                    VaeArchitectureError::MalformedConfiguration(
                        "batch-normalized VAE spatial ratio overflows".to_owned(),
                    )
                })?
            } else {
                base_ratio
            };
            let public_channels = selection.target_latent_channels.or(*embed_dim);
            Ok(spatial_ratio == latent.spatial_downscale_ratio
                && public_channels.is_none_or(|channels| channels == latent.channels))
        }
        VaeLoaderConfiguration::ExplicitAutoencoderKl { params_json, .. } => {
            let topology = ExplicitAutoencoderKlTopology::parse(params_json)?;
            Ok(topology.public_latent_channels()? == latent.channels)
        }
        _ => Ok(true),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimarySignal {
    Temporal,
    Taesd,
    StageA,
    StageCEncoder,
    StageCPreviewer,
    StageCCombined,
    DecoderConvIn,
    AudioOobleck,
    Mochi,
    LtxVideo,
    Conv3d,
    Cosmos,
    Wan,
    Shape,
    MusicDcae,
    Pixel,
    MmAudio,
    TaeHv,
    LtxAudio,
    StableAudio3,
    Tripo,
}

struct DiffusersPreconversion {
    key_mapping: Vec<(String, String)>,
    conv3d: bool,
}

fn normalize_diffusers_probe(
    probe: &ModelProbe,
    cancellation: &CancellationToken,
) -> Result<(Option<ModelProbe>, Option<DiffusersPreconversion>), VaeArchitectureError> {
    if !probe.tensor_shapes().contains_key(VAE_DIFFUSERS_SENTINEL) {
        return Ok((None, None));
    }
    cancellation.check()?;
    let conv3d = probe
        .tensor_shapes()
        .iter()
        .any(|(name, shape)| name.ends_with(".conv.weight") && shape.len() == 5);
    let mut tensor_shapes = std::collections::BTreeMap::new();
    let mut dtype_renames = std::collections::BTreeMap::new();
    for (index, (source_name, source_shape)) in probe.tensor_shapes().iter().enumerate() {
        if index % 256 == 0 {
            cancellation.check()?;
        }
        if source_shape.contains(&0) {
            return Err(VaeArchitectureError::MalformedDiffusersTensor {
                tensor: source_name.clone(),
                detail: format!("invalid zero-sized shape {source_shape:?}"),
            });
        }
        let target_name = convert_diffusers_vae_key(source_name);
        let mut target_shape = source_shape.clone();
        if ["q", "k", "v", "proj_out"]
            .iter()
            .any(|weight| target_name.contains(&format!("mid.attn_1.{weight}.weight")))
        {
            if target_shape.len() != 2 {
                return Err(VaeArchitectureError::MalformedDiffusersTensor {
                    tensor: source_name.clone(),
                    detail: format!("attention projection must have rank 2, got {target_shape:?}"),
                });
            }
            target_shape.extend(if conv3d {
                [1, 1, 1].as_slice()
            } else {
                [1, 1].as_slice()
            });
        }
        if tensor_shapes
            .insert(target_name.clone(), target_shape)
            .is_some()
        {
            return Err(VaeArchitectureError::DiffusersKeyCollision {
                target: target_name,
            });
        }
        dtype_renames.insert(source_name.as_str(), target_name);
    }
    let mut metadata = probe.metadata().clone();
    for (source_name, target_name) in &dtype_renames {
        let source_key = format!("{MODEL_PROBE_DTYPE_PREFIX}{source_name}");
        if let Some(dtype) = metadata.remove(&source_key) {
            let target_key = format!("{MODEL_PROBE_DTYPE_PREFIX}{target_name}");
            if metadata.insert(target_key.clone(), dtype).is_some() {
                return Err(VaeArchitectureError::DiffusersKeyCollision { target: target_key });
            }
        }
    }
    cancellation.check()?;
    let key_mapping = dtype_renames
        .iter()
        .map(|(source, target)| ((*source).to_owned(), target.clone()))
        .collect();
    Ok((
        Some(ModelProbe {
            tensor_shapes,
            metadata,
        }),
        Some(DiffusersPreconversion {
            key_mapping,
            conv3d,
        }),
    ))
}

fn convert_diffusers_vae_key(source: &str) -> String {
    let mut target = source
        .replace("conv_shortcut", "nin_shortcut")
        .replace("conv_norm_out", "norm_out")
        .replace("mid_block.attentions.0.", "mid.attn_1.");
    for block in 0..4 {
        for resnet in 0..2 {
            target = target.replace(
                &format!("encoder.down_blocks.{block}.resnets.{resnet}."),
                &format!("encoder.down.{block}.block.{resnet}."),
            );
        }
        if block < 3 {
            target = target.replace(
                &format!("down_blocks.{block}.downsamplers.0."),
                &format!("down.{block}.downsample."),
            );
            target = target.replace(
                &format!("up_blocks.{block}.upsamplers.0."),
                &format!("up.{}.upsample.", 3 - block),
            );
        }
        for resnet in 0..3 {
            target = target.replace(
                &format!("decoder.up_blocks.{block}.resnets.{resnet}."),
                &format!("decoder.up.{}.block.{resnet}.", 3 - block),
            );
        }
    }
    for block in 0..2 {
        target = target.replace(
            &format!("mid_block.resnets.{block}."),
            &format!("mid.block_{}.", block + 1),
        );
    }
    if source.contains("attentions") {
        for (diffusers, native) in [
            ("group_norm.", "norm."),
            ("query.", "q."),
            ("key.", "k."),
            ("value.", "v."),
            ("to_q.", "q."),
            ("to_k.", "k."),
            ("to_v.", "v."),
            ("to_out.0.", "proj_out."),
            ("proj_attn.", "proj_out."),
        ] {
            target = target.replace(diffusers, native);
        }
    }
    target
}

impl PrimarySignal {
    fn row_for_probe(self, probe: &ModelProbe) -> &'static str {
        match self {
            Self::Temporal => L504,
            Self::Taesd => L512,
            Self::StageA => L518,
            Self::StageCEncoder => L527,
            Self::StageCPreviewer => L535,
            Self::StageCCombined => L542,
            Self::DecoderConvIn => {
                probe
                    .tensor_shapes()
                    .get("decoder.conv_in.weight")
                    .map_or(L546, |shape| {
                        if shape.get(1) == Some(&64) {
                            L547
                        } else if shape.get(1) == Some(&32) && shape.len() == 5 {
                            L559
                        } else if probe.tensor_shapes().contains_key("post_quant_conv.weight") {
                            L603
                        } else {
                            L546
                        }
                    })
            }
            Self::AudioOobleck => L609,
            Self::Mochi => L636,
            Self::LtxVideo => L651,
            Self::Conv3d => probe
                .tensor_shapes()
                .get("decoder.conv_in.conv.weight")
                .map_or(L701, |shape| {
                    if shape.get(1) == Some(&32) {
                        L673
                    } else if probe
                        .tensor_shapes()
                        .contains_key("decoder.mid_block.resnets.0.norm1.norm_layer.weight")
                    {
                        L690
                    } else {
                        L701
                    }
                }),
            Self::Cosmos => L717,
            Self::Wan => {
                if probe
                    .tensor_shapes()
                    .contains_key("decoder.upsamples.0.upsamples.0.residual.2.weight")
                {
                    L731
                } else {
                    L730
                }
            }
            Self::Shape => L762,
            Self::MusicDcae => L784,
            Self::Pixel => L799,
            Self::MmAudio => L809,
            Self::TaeHv => match probe
                .tensor_shapes()
                .get("decoder.1.weight")
                .and_then(|shape| shape.get(1))
            {
                Some(48 | 128) => L835,
                Some(32)
                    if probe
                        .tensor_shapes()
                        .get("decoder.22.bias")
                        .and_then(|shape| shape.first())
                        == Some(&12) =>
                {
                    L840
                }
                _ => L828,
            },
            Self::LtxAudio => L856,
            Self::StableAudio3 => L874,
            Self::Tripo => L902,
        }
    }
}

fn primary_signals(probe: &ModelProbe) -> Vec<PrimarySignal> {
    let shapes = probe.tensor_shapes();
    let has = |key: &str| shapes.contains_key(key);
    let mut signals = Vec::new();
    let candidates = [
        (
            has("decoder.mid.block_1.mix_factor"),
            PrimarySignal::Temporal,
        ),
        (has("taesd_decoder.1.weight"), PrimarySignal::Taesd),
        (has("vquantizer.codebook.weight"), PrimarySignal::StageA),
        (
            has("backbone.1.0.block.0.1.num_batches_tracked"),
            PrimarySignal::StageCEncoder,
        ),
        (
            has("blocks.11.num_batches_tracked"),
            PrimarySignal::StageCPreviewer,
        ),
        (
            has("encoder.backbone.1.0.block.0.1.num_batches_tracked"),
            PrimarySignal::StageCCombined,
        ),
        (has("decoder.conv_in.weight"), PrimarySignal::DecoderConvIn),
        (
            has("decoder.layers.1.layers.0.beta"),
            PrimarySignal::AudioOobleck,
        ),
        (
            [
                "blocks.2.blocks.3.stack.5.weight",
                "decoder.blocks.2.blocks.3.stack.5.weight",
                "layers.4.layers.1.attn_block.attn.qkv.weight",
                "encoder.layers.4.layers.1.attn_block.attn.qkv.weight",
            ]
            .iter()
            .any(|key| has(key)),
            PrimarySignal::Mochi,
        ),
        (
            has("decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"),
            PrimarySignal::LtxVideo,
        ),
        (has("decoder.conv_in.conv.weight"), PrimarySignal::Conv3d),
        (has("decoder.unpatcher3d.wavelets"), PrimarySignal::Cosmos),
        (has("decoder.middle.0.residual.0.gamma"), PrimarySignal::Wan),
        (
            has("geo_decoder.cross_attn_decoder.ln_1.bias"),
            PrimarySignal::Shape,
        ),
        (
            has("vocoder.backbone.channel_layers.0.0.bias"),
            PrimarySignal::MusicDcae,
        ),
        (has("pixel_space_vae"), PrimarySignal::Pixel),
        (
            has("vocoder.activation_post.downsample.lowpass.filter"),
            PrimarySignal::MmAudio,
        ),
        (has("decoder.22.bias"), PrimarySignal::TaeHv),
        (
            has("vocoder.resblocks.0.convs1.0.weight")
                || has("vocoder.vocoder.resblocks.0.convs1.0.weight"),
            PrimarySignal::LtxAudio,
        ),
        (
            has("decoder.layers.3.transformers.0.pre_norm.alpha"),
            PrimarySignal::StableAudio3,
        ),
        (
            has("gs.base_offset_scale") || has("octree.out_proj.weight"),
            PrimarySignal::Tripo,
        ),
    ];
    for (present, signal) in candidates {
        if present {
            signals.push(signal);
        }
    }
    signals
}

fn select_signal(
    signal: PrimarySignal,
    probe: &ModelProbe,
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    match signal {
        PrimarySignal::Temporal => simple(
            probe,
            L504,
            "decoder.mid.block_1.mix_factor",
            "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1",
            VaeKernelProfile::TemporalAutoencodingEngineV1,
            Some(4),
            2,
            VaeBoundaryDomain::Image,
        ),
        PrimarySignal::Taesd => {
            let shape = require_rank(probe, L512, "taesd_decoder.1.weight", 2)?;
            let metadata_channels = probe
                .metadata()
                .get("tae_latent_channels")
                .map(|value| parse_metadata_u64("tae_latent_channels", value))
                .transpose()?;
            let latent_channels = metadata_channels.unwrap_or(shape[1]);
            let mut selection = selected(
                "sim.taesd.TAESD.v1",
                VaeKernelProfile::TaesdV1,
                vec![L512],
                Some(latent_channels),
                2,
                VaeBoundaryDomain::Image,
            )?;
            selection.loader_configuration = VaeLoaderConfiguration::Taesd {
                latent_channels,
                metadata_override: metadata_channels.is_some(),
            };
            Ok(selection)
        }
        PrimarySignal::StageA => simple(
            probe,
            L518,
            "vquantizer.codebook.weight",
            "sim.ldm.cascade.stage_a.StageA.v1",
            VaeKernelProfile::StableCascadeStageAV1,
            Some(4),
            2,
            VaeBoundaryDomain::Image,
        ),
        PrimarySignal::StageCEncoder => simple(
            probe,
            L527,
            "backbone.1.0.block.0.1.num_batches_tracked",
            "sim.ldm.cascade.stage_c.StageCEncoder.v1",
            VaeKernelProfile::StableCascadeStageCEncoderV1,
            Some(16),
            2,
            VaeBoundaryDomain::Image,
        ),
        PrimarySignal::StageCPreviewer => simple(
            probe,
            L535,
            "blocks.11.num_batches_tracked",
            "sim.ldm.cascade.stage_c.StageCPreviewer.v1",
            VaeKernelProfile::StableCascadeStageCPreviewerV1,
            Some(16),
            2,
            VaeBoundaryDomain::Image,
        ),
        PrimarySignal::StageCCombined => simple(
            probe,
            L542,
            "encoder.backbone.1.0.block.0.1.num_batches_tracked",
            "sim.ldm.cascade.stage_c.StageCCombined.v1",
            VaeKernelProfile::StableCascadeStageCCombinedV1,
            Some(16),
            2,
            VaeBoundaryDomain::Image,
        ),
        PrimarySignal::DecoderConvIn => select_decoder_conv_in(probe),
        PrimarySignal::AudioOobleck => select_oobleck(probe),
        PrimarySignal::Mochi => select_mochi(probe),
        PrimarySignal::LtxVideo => select_ltx_video(probe),
        PrimarySignal::Conv3d => select_conv3d(probe),
        PrimarySignal::Cosmos => simple(
            probe,
            L717,
            "decoder.unpatcher3d.wavelets",
            "sim.ldm.cosmos.vae.CausalContinuousVideoTokenizer.v1",
            VaeKernelProfile::CosmosV1,
            Some(16),
            3,
            VaeBoundaryDomain::Video,
        ),
        PrimarySignal::Wan => select_wan(probe),
        PrimarySignal::Shape => simple(
            probe,
            L762,
            "geo_decoder.cross_attn_decoder.ln_1.bias",
            "sim.ldm.hunyuan3d.vae.ShapeVAE.v1",
            VaeKernelProfile::HunyuanShapeV1,
            None,
            1,
            VaeBoundaryDomain::Structured,
        ),
        PrimarySignal::MusicDcae => simple(
            probe,
            L784,
            "vocoder.backbone.channel_layers.0.0.bias",
            "sim.ldm.ace.vae.MusicDCAE.v1",
            VaeKernelProfile::MusicDcaeV1,
            Some(8),
            2,
            VaeBoundaryDomain::Audio,
        ),
        PrimarySignal::Pixel => select_pixel(probe),
        PrimarySignal::MmAudio => simple(
            probe,
            L809,
            "vocoder.activation_post.downsample.lowpass.filter",
            "sim.ldm.mmaudio.vae.AudioAutoencoder.v1",
            VaeKernelProfile::MmAudio16KhzV1,
            Some(20),
            1,
            VaeBoundaryDomain::Audio,
        ),
        PrimarySignal::TaeHv => select_taehv(probe),
        PrimarySignal::LtxAudio => select_ltx_audio(probe),
        PrimarySignal::StableAudio3 => select_stable_audio3(probe),
        PrimarySignal::Tripo => select_tripo(probe),
    }
}

fn select_decoder_conv_in(
    probe: &ModelProbe,
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let shape = require_rank_allow_zero(probe, L546, "decoder.conv_in.weight", 2)?;
    if shape[1] == 64 {
        validate_nonzero_shape(L547, "decoder.conv_in.weight", shape)?;
        return selected(
            "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1",
            VaeKernelProfile::HunyuanImageV1,
            vec![L546, L547],
            Some(64),
            2,
            VaeBoundaryDomain::Image,
        );
    }
    if shape[1] == 32 && shape.len() == 5 {
        validate_nonzero_shape(L559, "decoder.conv_in.weight", shape)?;
        let mut selection = selected(
            "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.image.v1",
            VaeKernelProfile::HunyuanImageRefinerV1,
            vec![L546, L559],
            Some(32),
            3,
            VaeBoundaryDomain::Image,
        )?;
        selection.target_latent_channels = Some(64);
        return Ok(selection);
    }
    validate_nonzero_shape(L546, "decoder.conv_in.weight", shape)?;
    let post_quant_key = if probe.tensor_shapes().contains_key("post_quant_conv.weight") {
        Some("post_quant_conv.weight")
    } else if probe
        .tensor_shapes()
        .contains_key("decoder.post_quant_conv.weight")
    {
        Some("decoder.post_quant_conv.weight")
    } else {
        None
    };
    let autoencoder_kl = post_quant_key.is_some();
    if let Some(post_quant_key) = post_quant_key {
        require_rank(probe, L603, post_quant_key, 2)?;
    }
    let x4 = !probe
        .tensor_shapes()
        .contains_key("encoder.down.2.downsample.conv.weight")
        && !probe
            .tensor_shapes()
            .contains_key("decoder.up.3.upsample.conv.weight");
    let batch_norm_latent = probe.tensor_shapes().contains_key("bn.running_mean");
    let profile = match (autoencoder_kl, x4, batch_norm_latent) {
        (true, _, true) => VaeKernelProfile::AutoencoderKlBatchNormV1,
        (false, _, true) => VaeKernelProfile::AutoencodingEngineBatchNormV1,
        (true, true, false) => VaeKernelProfile::AutoencoderKlX4V1,
        (false, true, false) => VaeKernelProfile::AutoencodingEngineX4V1,
        (true, false, false) => VaeKernelProfile::AutoencoderKlV1,
        (false, false, false) => VaeKernelProfile::AutoencodingEngineV1,
    };
    let mut trace = vec![L546];
    if autoencoder_kl {
        trace.push(L603);
    }
    let mut selection = selected(
        "sim.ldm.models.autoencoder.AutoencoderKL.v1",
        profile,
        trace,
        Some(shape[1]),
        2,
        VaeBoundaryDomain::Image,
    )?;
    let legacy_prefix_rewrite = probe
        .tensor_shapes()
        .contains_key("decoder.post_quant_conv.weight");
    let channel_multiplier = if x4 { 4 } else { 4 };
    let decoder_channels = shape[0] / channel_multiplier;
    let asymmetric_decoder_channels = (decoder_channels != 128).then_some(decoder_channels);
    let embed_dim = probe
        .tensor_shapes()
        .get(post_quant_key.unwrap_or("post_quant_conv.weight"))
        .and_then(|shape| shape.get(1))
        .copied();
    if autoencoder_kl {
        selection.latent_channels = embed_dim.or(selection.latent_channels);
        selection.target_latent_channels = selection.latent_channels;
    }
    if batch_norm_latent {
        selection.latent_channels = selection
            .latent_channels
            .and_then(|value| value.checked_mul(4));
        selection.target_latent_channels = selection.latent_channels;
    }
    selection.loader_configuration = VaeLoaderConfiguration::DefaultKl {
        x4,
        legacy_prefix_rewrite,
        batch_norm_latent,
        asymmetric_decoder_channels,
        embed_dim,
    };
    Ok(selection)
}

fn select_oobleck(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    require_marker(probe, L609, "decoder.layers.1.layers.0.beta")?;
    let parameter_shape = [
        "decoder.layers.2.layers.1.parametrizations.weight.original1",
        "decoder.layers.2.layers.1.weight_v",
    ]
    .iter()
    .find_map(|key| probe.tensor_shapes().get(*key));
    if parameter_shape.is_some_and(|shape| shape.is_empty() || shape.contains(&0)) {
        return Err(VaeArchitectureError::Partial {
            row: L609,
            detail: "Oobleck parametrized weight has an invalid shape".to_owned(),
        });
    }
    let forty_eight_khz = parameter_shape.is_some_and(|shape| shape.last() == Some(&12));
    selected(
        "sim.ldm.audio.vae.AudioOobleckVAE.v1",
        if forty_eight_khz {
            VaeKernelProfile::AudioOobleck48KhzV1
        } else {
            VaeKernelProfile::AudioOobleck44KhzV1
        },
        vec![L609],
        Some(64),
        1,
        VaeBoundaryDomain::Audio,
    )
}

fn select_mochi(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let key = [
        "blocks.2.blocks.3.stack.5.weight",
        "decoder.blocks.2.blocks.3.stack.5.weight",
        "layers.4.layers.1.attn_block.attn.qkv.weight",
        "encoder.layers.4.layers.1.attn_block.attn.qkv.weight",
    ]
    .iter()
    .find(|key| probe.tensor_shapes().contains_key(**key))
    .copied()
    .ok_or_else(|| VaeArchitectureError::Partial {
        row: L636,
        detail: "no complete Mochi sentinel".to_owned(),
    })?;
    require_marker(probe, L636, key)?;
    selected(
        "sim.ldm.genmo.vae.VideoVAE.v1",
        VaeKernelProfile::MochiV1,
        vec![L636],
        Some(12),
        3,
        VaeBoundaryDomain::Video,
    )
}

fn select_ltx_video(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let shape = require_rank(
        probe,
        L651,
        "decoder.up_blocks.0.res_blocks.0.conv1.conv.weight",
        1,
    )?;
    let (configuration_sha256, configuration_json) = ltx_configuration(probe)?;
    let profile = match shape[0] {
        512 => VaeKernelProfile::LtxVideoV0 {
            configuration_sha256,
        },
        1024 if probe
            .tensor_shapes()
            .contains_key("encoder.down_blocks.1.conv.conv.bias") =>
        {
            VaeKernelProfile::LtxVideoV2 {
                configuration_sha256,
            }
        }
        1024 => VaeKernelProfile::LtxVideoV1 {
            configuration_sha256,
        },
        _ => VaeKernelProfile::LtxVideoV0 {
            configuration_sha256,
        },
    };
    let mut selection = selected(
        "sim.ldm.lightricks.vae.VideoVAE.v1",
        profile,
        vec![L651],
        Some(128),
        3,
        VaeBoundaryDomain::Video,
    )?;
    selection.loader_configuration = VaeLoaderConfiguration::LtxVideo {
        configuration_sha256: match selection.profile() {
            VaeKernelProfile::LtxVideoV0 {
                configuration_sha256,
            }
            | VaeKernelProfile::LtxVideoV1 {
                configuration_sha256,
            }
            | VaeKernelProfile::LtxVideoV2 {
                configuration_sha256,
            } => configuration_sha256.clone(),
            _ => None,
        },
        configuration_json,
    };
    Ok(selection)
}

fn ltx_configuration(
    probe: &ModelProbe,
) -> Result<(Option<String>, Option<String>), VaeArchitectureError> {
    let Some(encoded) = probe.metadata().get("config") else {
        return Ok((None, None));
    };
    if encoded.len() > MAX_VAE_CONFIGURATION_BYTES {
        return Err(VaeArchitectureError::ConfigurationLimit {
            actual: encoded.len(),
            maximum: MAX_VAE_CONFIGURATION_BYTES,
        });
    }
    let root: serde_json::Value =
        serde_json::from_str(encoded).map_err(|error| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: error.to_string(),
        })?;
    let object = root
        .as_object()
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "expected a JSON object".to_owned(),
        })?;
    let Some(vae) = object.get("vae") else {
        return Ok((None, None));
    };
    if !vae.is_object() {
        return Err(VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "vae must be an object".to_owned(),
        });
    }
    let configuration_json = canonical_json(vae)?;
    Ok((
        Some(format!(
            "{:x}",
            Sha256::digest(configuration_json.as_bytes())
        )),
        Some(configuration_json),
    ))
}

struct LtxAudioConfiguration {
    autoencoder_sha256: String,
    autoencoder_json: String,
    vocoder_sha256: String,
    vocoder_json: String,
    latent_channels: u64,
    input_sample_rate: u32,
    output_sample_rate: u32,
}

fn ltx_audio_configuration(
    probe: &ModelProbe,
) -> Result<LtxAudioConfiguration, VaeArchitectureError> {
    let encoded =
        probe
            .metadata()
            .get("config")
            .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: "LTX Audio requires metadata config".to_owned(),
            })?;
    if encoded.len() > MAX_VAE_CONFIGURATION_BYTES {
        return Err(VaeArchitectureError::ConfigurationLimit {
            actual: encoded.len(),
            maximum: MAX_VAE_CONFIGURATION_BYTES,
        });
    }
    let root: serde_json::Value =
        serde_json::from_str(encoded).map_err(|error| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: error.to_string(),
        })?;
    let object = root
        .as_object()
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "expected a JSON object".to_owned(),
        })?;
    let normalized = |key: &'static str| -> Result<(String, String), VaeArchitectureError> {
        let value = object
            .get(key)
            .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: format!("missing {key} object"),
            })?;
        if !value.is_object() {
            return Err(VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: format!("{key} must be an object"),
            });
        }
        let json = canonical_json(value)?;
        Ok((format!("{:x}", Sha256::digest(json.as_bytes())), json))
    };
    let autoencoder = object
        .get("audio_vae")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "audio_vae must be an object".to_owned(),
        })?;
    let vocoder = object
        .get("vocoder")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "vocoder must be an object".to_owned(),
        })?;
    let model_params = autoencoder
        .get("model")
        .and_then(serde_json::Value::as_object)
        .and_then(|model| model.get("params"))
        .and_then(serde_json::Value::as_object);
    let codec = model_params
        .and_then(|params| {
            params
                .get("decoder")
                .or_else(|| params.get("encoder"))
                .or_else(|| params.get("ddconfig"))
        })
        .and_then(serde_json::Value::as_object);
    let latent_channels = required_configuration_u64(codec, "z_channels")?;
    let input_sample_rate = optional_configuration_u64(model_params, "sampling_rate")?
        .or(optional_configuration_u64(
            Some(autoencoder),
            "sampling_rate",
        )?)
        .unwrap_or(16_000);
    let stft = autoencoder
        .get("preprocessing")
        .and_then(serde_json::Value::as_object)
        .and_then(|preprocessing| preprocessing.get("stft"))
        .and_then(serde_json::Value::as_object);
    let hop_length = optional_configuration_u64(stft, "hop_length")?.unwrap_or(160);
    let output_sample_rate = if let Some(bwe) =
        vocoder.get("bwe").and_then(serde_json::Value::as_object)
    {
        required_configuration_u64(Some(bwe), "output_sampling_rate")?
    } else if let Some(output) = optional_configuration_u64(Some(vocoder), "output_sample_rate")? {
        output
    } else {
        let upsample_factor = match vocoder
            .get("upsample_rates")
            .and_then(serde_json::Value::as_array)
        {
            Some(rates) => rates
                .iter()
                .try_fold(1_u64, |product, rate| product.checked_mul(rate.as_u64()?)),
            None => Some(160),
        }
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "vocoder upsample rates must be positive integers without overflow".to_owned(),
        })?;
        input_sample_rate
            .checked_mul(upsample_factor)
            .and_then(|rate| rate.checked_div(hop_length))
            .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: "audio sample-rate inference overflowed or divided by zero".to_owned(),
            })?
    };
    if latent_channels == 0 || input_sample_rate == 0 || output_sample_rate == 0 {
        return Err(VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: "audio channels and sample rates must be positive".to_owned(),
        });
    }
    let (autoencoder_sha256, autoencoder_json) = normalized("audio_vae")?;
    let (vocoder_sha256, vocoder_json) = normalized("vocoder")?;
    Ok(LtxAudioConfiguration {
        autoencoder_sha256,
        autoencoder_json,
        vocoder_sha256,
        vocoder_json,
        latent_channels,
        input_sample_rate: u32::try_from(input_sample_rate).map_err(|_| {
            VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: "input sample rate exceeds u32".to_owned(),
            }
        })?,
        output_sample_rate: u32::try_from(output_sample_rate).map_err(|_| {
            VaeArchitectureError::MalformedMetadata {
                key: "config",
                detail: "output sample rate exceeds u32".to_owned(),
            }
        })?,
    })
}

fn optional_configuration_u64(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &'static str,
) -> Result<Option<u64>, VaeArchitectureError> {
    let Some(value) = object.and_then(|object| object.get(key)) else {
        return Ok(None);
    };
    value
        .as_u64()
        .map(Some)
        .ok_or_else(|| VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: format!("{key} must be an unsigned integer"),
        })
}

fn required_configuration_u64(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &'static str,
) -> Result<u64, VaeArchitectureError> {
    optional_configuration_u64(object, key)?.ok_or_else(|| {
        VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: format!("{key} is required"),
        }
    })
}

fn canonical_json(value: &serde_json::Value) -> Result<String, VaeArchitectureError> {
    fn normalize(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(object) => {
                let mut keys = object.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                let mut normalized = serde_json::Map::new();
                for key in keys {
                    if let Some(value) = object.get(key) {
                        normalized.insert(key.clone(), normalize(value));
                    }
                }
                serde_json::Value::Object(normalized)
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(normalize).collect())
            }
            value => value.clone(),
        }
    }
    serde_json::to_string(&normalize(value)).map_err(|error| {
        VaeArchitectureError::MalformedMetadata {
            key: "config",
            detail: error.to_string(),
        }
    })
}

fn parse_metadata_u64(key: &'static str, value: &str) -> Result<u64, VaeArchitectureError> {
    let parsed = value
        .parse::<u64>()
        .or_else(|_| serde_json::from_str::<u64>(value))
        .map_err(|error| VaeArchitectureError::MalformedMetadata {
            key,
            detail: error.to_string(),
        })?;
    if parsed == 0 {
        return Err(VaeArchitectureError::MalformedMetadata {
            key,
            detail: "expected a positive channel count".to_owned(),
        });
    }
    Ok(parsed)
}

fn select_conv3d(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let shape = require_rank_allow_zero(probe, L701, "decoder.conv_in.conv.weight", 2)?;
    if shape[1] == 32 {
        validate_nonzero_shape(L673, "decoder.conv_in.conv.weight", shape)?;
        return selected(
            "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.video.v1",
            VaeKernelProfile::HunyuanVideoRefinerV1,
            vec![L673],
            Some(32),
            3,
            VaeBoundaryDomain::Video,
        );
    }
    if probe
        .tensor_shapes()
        .contains_key("decoder.mid_block.resnets.0.norm1.norm_layer.weight")
    {
        validate_nonzero_shape(L690, "decoder.conv_in.conv.weight", shape)?;
        let output = require_rank(probe, L690, "encoder.conv_out.conv.weight", 1)?;
        if !output[0].is_multiple_of(2) || output[0] == 0 {
            return Err(VaeArchitectureError::Partial {
                row: L690,
                detail: "encoder output channels must be positive and even".to_owned(),
            });
        }
        return selected(
            "sim.ldm.cogvideo.vae.AutoencoderKLCogVideoX.v1",
            VaeKernelProfile::CogVideoXV1,
            vec![L690],
            Some(output[0] / 2),
            3,
            VaeBoundaryDomain::Video,
        );
    }
    validate_nonzero_shape(L701, "decoder.conv_in.conv.weight", shape)?;
    require_rank(probe, L701, "post_quant_conv.weight", 2)?;
    selected(
        "sim.ldm.models.autoencoder.AutoencoderKL.causal3d.v1",
        VaeKernelProfile::Causal3dV1,
        vec![L701],
        Some(shape[1]),
        3,
        VaeBoundaryDomain::Video,
    )
}

fn select_wan(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    require_rank(probe, L730, "decoder.middle.0.residual.0.gamma", 1)?;
    if probe
        .tensor_shapes()
        .contains_key("decoder.upsamples.0.upsamples.0.residual.2.weight")
    {
        require_marker(
            probe,
            L731,
            "decoder.upsamples.0.upsamples.0.residual.2.weight",
        )?;
        return selected(
            "sim.ldm.wan.vae2_2.WanVAE.v1",
            VaeKernelProfile::Wan22V1,
            vec![L730, L731],
            Some(48),
            3,
            VaeBoundaryDomain::Video,
        );
    }
    require_rank(probe, L730, "decoder.head.0.gamma", 1)?;
    require_rank(probe, L730, "encoder.conv1.weight", 2)?;
    require_rank(probe, L730, "decoder.head.2.weight", 1)?;
    selected(
        "sim.ldm.wan.vae.WanVAE.v1",
        VaeKernelProfile::Wan21V1,
        vec![L730],
        Some(16),
        3,
        VaeBoundaryDomain::Video,
    )
}

fn select_taehv(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let bias = require_rank_allow_zero(probe, L828, "decoder.22.bias", 1)?;
    let decoder = require_rank_allow_zero(probe, L828, "decoder.1.weight", 2)?;
    let channels = decoder[1];
    let (profile, trace) = match channels {
        48 => (VaeKernelProfile::TaeHvWan22V1, vec![L828, L835]),
        128 => (VaeKernelProfile::TaeHvLtx2V1, vec![L828, L835]),
        32 if bias[0] == 12 => (VaeKernelProfile::LightTaeHv15V1, vec![L828, L840]),
        _ if matches!(
            probe.storage_dtype("decoder.1.weight"),
            Some(ModelStorageDType::Tensor(DType::F16))
        ) =>
        {
            (VaeKernelProfile::TaeHvHunyuanV1, vec![L828])
        }
        _ => (VaeKernelProfile::LightTaeWan21V1, vec![L828]),
    };
    let selected_row = trace.last().copied().unwrap_or(L828);
    validate_nonzero_shape(selected_row, "decoder.22.bias", bias)?;
    validate_nonzero_shape(selected_row, "decoder.1.weight", decoder)?;
    selected(
        "sim.taesd.taehv.TAEHV.v1",
        profile,
        trace,
        Some(channels),
        3,
        VaeBoundaryDomain::Video,
    )
}

fn select_pixel(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let shape = probe
        .tensor_shapes()
        .get("pixel_space_vae")
        .ok_or_else(|| VaeArchitectureError::Partial {
            row: L799,
            detail: "missing pixel-space marker".to_owned(),
        })?;
    if shape.contains(&0) {
        return Err(VaeArchitectureError::Partial {
            row: L799,
            detail: format!("pixel-space marker has invalid shape {shape:?}"),
        });
    }
    selected(
        "sim.pixel_space_convert.PixelspaceConversionVAE.v1",
        VaeKernelProfile::PixelSpaceV1,
        vec![L799],
        Some(3),
        2,
        VaeBoundaryDomain::Image,
    )
}

fn select_ltx_audio(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let key = [
        "vocoder.resblocks.0.convs1.0.weight",
        "vocoder.vocoder.resblocks.0.convs1.0.weight",
    ]
    .iter()
    .find(|key| probe.tensor_shapes().contains_key(**key))
    .copied()
    .ok_or_else(|| VaeArchitectureError::Partial {
        row: L856,
        detail: "no complete LTX audio sentinel".to_owned(),
    })?;
    require_marker(probe, L856, key)?;
    let configuration = ltx_audio_configuration(probe)?;
    let mut selection = selected(
        "sim.ldm.lightricks.vae.audio_vae.AudioVAE.v1",
        VaeKernelProfile::LtxAudioV1,
        vec![L856],
        Some(configuration.latent_channels),
        2,
        VaeBoundaryDomain::Audio,
    )?;
    selection.loader_configuration = VaeLoaderConfiguration::LtxAudio {
        autoencoder_sha256: configuration.autoencoder_sha256,
        autoencoder_json: configuration.autoencoder_json,
        vocoder_sha256: configuration.vocoder_sha256,
        vocoder_json: configuration.vocoder_json,
        latent_channels: configuration.latent_channels,
        input_sample_rate: configuration.input_sample_rate,
        output_sample_rate: configuration.output_sample_rate,
    };
    Ok(selection)
}

fn select_stable_audio3(
    probe: &ModelProbe,
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    require_marker(
        probe,
        L874,
        "decoder.layers.3.transformers.0.pre_norm.alpha",
    )?;
    let profile = if probe
        .tensor_shapes()
        .contains_key("decoder.layers.3.transformers.11.self_attn.to_out.weight")
    {
        VaeKernelProfile::StableAudio3DeepV1
    } else {
        VaeKernelProfile::StableAudio3ShallowV1
    };
    selected(
        "sim.ldm.audio.vae_sa3.SA3AudioVAE.v1",
        profile,
        vec![L874],
        Some(256),
        1,
        VaeBoundaryDomain::Audio,
    )
}

fn select_tripo(probe: &ModelProbe) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    require_marker(probe, L902, "gs.base_offset_scale")?;
    require_rank(probe, L902, "octree.out_proj.weight", 1)?;
    selected(
        "sim.ldm.triposplat.vae.OctreeGaussianDecoder.v1",
        VaeKernelProfile::TripoSplatV1,
        vec![L902],
        Some(16),
        1,
        VaeBoundaryDomain::Structured,
    )
}

#[allow(clippy::too_many_arguments)]
fn simple(
    probe: &ModelProbe,
    row: &'static str,
    key: &'static str,
    architecture: &'static str,
    profile: VaeKernelProfile,
    latent_channels: Option<u64>,
    latent_dimensions: u8,
    boundary: VaeBoundaryDomain,
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    require_marker(probe, row, key)?;
    selected(
        architecture,
        profile,
        vec![row],
        latent_channels,
        latent_dimensions,
        boundary,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected(
    architecture: &'static str,
    profile: VaeKernelProfile,
    trace: Vec<&'static str>,
    latent_channels: Option<u64>,
    latent_dimensions: u8,
    boundary: VaeBoundaryDomain,
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let contract = profile.contract();
    let expected_boundary = match contract.boundary {
        crate::VaeBoundaryKind::Image => VaeBoundaryDomain::Image,
        crate::VaeBoundaryKind::Video => VaeBoundaryDomain::Video,
        crate::VaeBoundaryKind::Audio => VaeBoundaryDomain::Audio,
        crate::VaeBoundaryKind::StructuredOutput => VaeBoundaryDomain::Structured,
    };
    if latent_dimensions != contract.latent_dimensions || boundary != expected_boundary {
        return Err(VaeArchitectureError::Registry(format!(
            "selector geometry diverges from canonical profile contract for {profile:?}"
        )));
    }
    selected_with(
        architecture,
        profile,
        trace,
        latent_channels,
        contract.target_latent_channels.or(latent_channels),
        latent_dimensions,
        boundary,
        contract.canonical_compatibility,
        VaeLoaderConfiguration::Automatic,
        contract.supported_dtypes,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected_with(
    architecture: &'static str,
    profile: VaeKernelProfile,
    trace: Vec<&'static str>,
    latent_channels: Option<u64>,
    target_latent_channels: Option<u64>,
    latent_dimensions: u8,
    boundary: VaeBoundaryDomain,
    canonical_compatibility: VaeCanonicalCompatibility,
    loader_configuration: VaeLoaderConfiguration,
    supported_dtypes: &'static [DType],
) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
    let supported_devices = if boundary == VaeBoundaryDomain::Image {
        IMAGE_NATIVE_DEVICES
    } else {
        &DeviceKind::ALL
    };
    Ok(VaeArchitectureSelection {
        architecture: VaeArchitectureIdentity::checked(architecture)
            .map_err(|error| VaeArchitectureError::Registry(error.to_string()))?,
        profile,
        trace,
        latent_channels,
        target_latent_channels,
        latent_dimensions,
        boundary,
        canonical_compatibility,
        loader_configuration,
        supported_dtypes,
        supported_devices,
    })
}

fn require_marker<'a>(
    probe: &'a ModelProbe,
    row: &'static str,
    key: &'static str,
) -> Result<&'a [u64], VaeArchitectureError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| VaeArchitectureError::Partial {
            row,
            detail: format!("missing required tensor {key}"),
        })?;
    if shape.contains(&0) {
        return Err(VaeArchitectureError::Partial {
            row,
            detail: format!("tensor {key} has invalid shape {shape:?}"),
        });
    }
    Ok(shape)
}

fn require_rank<'a>(
    probe: &'a ModelProbe,
    row: &'static str,
    key: &'static str,
    minimum_rank: usize,
) -> Result<&'a [u64], VaeArchitectureError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| VaeArchitectureError::Partial {
            row,
            detail: format!("missing required tensor {key}"),
        })?;
    if shape.len() < minimum_rank || shape.contains(&0) {
        return Err(VaeArchitectureError::Partial {
            row,
            detail: format!("tensor {key} has invalid shape {shape:?}"),
        });
    }
    Ok(shape)
}

fn require_rank_allow_zero<'a>(
    probe: &'a ModelProbe,
    row: &'static str,
    key: &'static str,
    minimum_rank: usize,
) -> Result<&'a [u64], VaeArchitectureError> {
    let shape = probe
        .tensor_shapes()
        .get(key)
        .ok_or_else(|| VaeArchitectureError::Partial {
            row,
            detail: format!("missing required tensor {key}"),
        })?;
    if shape.len() < minimum_rank {
        return Err(VaeArchitectureError::Partial {
            row,
            detail: format!("tensor {key} has invalid shape {shape:?}"),
        });
    }
    Ok(shape)
}

fn validate_nonzero_shape(
    row: &'static str,
    key: &'static str,
    shape: &[u64],
) -> Result<(), VaeArchitectureError> {
    if shape.contains(&0) {
        Err(VaeArchitectureError::Partial {
            row,
            detail: format!("tensor {key} has invalid shape {shape:?}"),
        })
    } else {
        Ok(())
    }
}

pub(crate) fn is_registered_vae_architecture(value: &str) -> bool {
    matches!(
        value,
        "zed.vae.block_average_nearest.v1"
            | "zed.vae.boundary.v1"
            | "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1"
            | "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1"
            | "sim.taesd.TAESD.v1"
            | "sim.ldm.cascade.stage_a.StageA.v1"
            | "sim.ldm.cascade.stage_c.StageCEncoder.v1"
            | "sim.ldm.cascade.stage_c.StageCPreviewer.v1"
            | "sim.ldm.cascade.stage_c.StageCCombined.v1"
            | "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1"
            | "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.image.v1"
            | "sim.ldm.models.autoencoder.AutoencoderKL.v1"
            | "sim.ldm.audio.vae.AudioOobleckVAE.v1"
            | "sim.ldm.genmo.vae.VideoVAE.v1"
            | "sim.ldm.lightricks.vae.VideoVAE.v1"
            | "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.video.v1"
            | "sim.ldm.cogvideo.vae.AutoencoderKLCogVideoX.v1"
            | "sim.ldm.models.autoencoder.AutoencoderKL.causal3d.v1"
            | "sim.ldm.cosmos.vae.CausalContinuousVideoTokenizer.v1"
            | "sim.ldm.wan.vae2_2.WanVAE.v1"
            | "sim.ldm.wan.vae.WanVAE.v1"
            | "sim.ldm.hunyuan3d.vae.ShapeVAE.v1"
            | "sim.ldm.ace.vae.MusicDCAE.v1"
            | "sim.pixel_space_convert.PixelspaceConversionVAE.v1"
            | "sim.ldm.mmaudio.vae.AudioAutoencoder.v1"
            | "sim.taesd.taehv.TAEHV.v1"
            | "sim.ldm.lightricks.vae.audio_vae.AudioVAE.v1"
            | "sim.ldm.audio.vae_sa3.SA3AudioVAE.v1"
            | "sim.ldm.triposplat.vae.OctreeGaussianDecoder.v1"
    )
}

pub(crate) fn validate_architecture_profile_pair(
    architecture: &VaeArchitectureIdentity,
    profile: &VaeKernelProfile,
) -> Result<(), VaeArchitectureError> {
    let matches = match architecture.as_str() {
        "zed.vae.block_average_nearest.v1" | "zed.vae.boundary.v1" => {
            matches!(profile, VaeKernelProfile::BlockAverageNearestV1)
        }
        "sim.ldm.models.autoencoder.AutoencoderKL.reduced.v1" => {
            matches!(profile, VaeKernelProfile::Sd15AutoencoderKlReducedV1)
        }
        "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1" => {
            matches!(profile, VaeKernelProfile::TemporalAutoencodingEngineV1)
        }
        "sim.taesd.TAESD.v1" => matches!(profile, VaeKernelProfile::TaesdV1),
        "sim.ldm.cascade.stage_a.StageA.v1" => {
            matches!(profile, VaeKernelProfile::StableCascadeStageAV1)
        }
        "sim.ldm.cascade.stage_c.StageCEncoder.v1" => {
            matches!(profile, VaeKernelProfile::StableCascadeStageCEncoderV1)
        }
        "sim.ldm.cascade.stage_c.StageCPreviewer.v1" => {
            matches!(profile, VaeKernelProfile::StableCascadeStageCPreviewerV1)
        }
        "sim.ldm.cascade.stage_c.StageCCombined.v1" => {
            matches!(profile, VaeKernelProfile::StableCascadeStageCCombinedV1)
        }
        "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1" => {
            matches!(profile, VaeKernelProfile::HunyuanImageV1)
        }
        "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.image.v1" => {
            matches!(profile, VaeKernelProfile::HunyuanImageRefinerV1)
        }
        "sim.ldm.models.autoencoder.AutoencoderKL.v1" => matches!(
            profile,
            VaeKernelProfile::AutoencoderKlV1
                | VaeKernelProfile::AutoencoderKlX4V1
                | VaeKernelProfile::AutoencoderKlBatchNormV1
                | VaeKernelProfile::ExplicitAutoencoderKlV1
                | VaeKernelProfile::AutoencodingEngineV1
                | VaeKernelProfile::AutoencodingEngineX4V1
                | VaeKernelProfile::AutoencodingEngineBatchNormV1
        ),
        "sim.ldm.audio.vae.AudioOobleckVAE.v1" => matches!(
            profile,
            VaeKernelProfile::AudioOobleck44KhzV1 | VaeKernelProfile::AudioOobleck48KhzV1
        ),
        "sim.ldm.genmo.vae.VideoVAE.v1" => matches!(profile, VaeKernelProfile::MochiV1),
        "sim.ldm.lightricks.vae.VideoVAE.v1" => matches!(
            profile,
            VaeKernelProfile::LtxVideoV0 { .. }
                | VaeKernelProfile::LtxVideoV1 { .. }
                | VaeKernelProfile::LtxVideoV2 { .. }
        ),
        "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.video.v1" => {
            matches!(profile, VaeKernelProfile::HunyuanVideoRefinerV1)
        }
        "sim.ldm.cogvideo.vae.AutoencoderKLCogVideoX.v1" => {
            matches!(profile, VaeKernelProfile::CogVideoXV1)
        }
        "sim.ldm.models.autoencoder.AutoencoderKL.causal3d.v1" => {
            matches!(profile, VaeKernelProfile::Causal3dV1)
        }
        "sim.ldm.cosmos.vae.CausalContinuousVideoTokenizer.v1" => {
            matches!(profile, VaeKernelProfile::CosmosV1)
        }
        "sim.ldm.wan.vae2_2.WanVAE.v1" => matches!(profile, VaeKernelProfile::Wan22V1),
        "sim.ldm.wan.vae.WanVAE.v1" => matches!(profile, VaeKernelProfile::Wan21V1),
        "sim.ldm.hunyuan3d.vae.ShapeVAE.v1" => {
            matches!(profile, VaeKernelProfile::HunyuanShapeV1)
        }
        "sim.ldm.ace.vae.MusicDCAE.v1" => matches!(profile, VaeKernelProfile::MusicDcaeV1),
        "sim.pixel_space_convert.PixelspaceConversionVAE.v1" => {
            matches!(profile, VaeKernelProfile::PixelSpaceV1)
        }
        "sim.ldm.mmaudio.vae.AudioAutoencoder.v1" => {
            matches!(profile, VaeKernelProfile::MmAudio16KhzV1)
        }
        "sim.taesd.taehv.TAEHV.v1" => matches!(
            profile,
            VaeKernelProfile::TaeHvWan22V1
                | VaeKernelProfile::TaeHvLtx2V1
                | VaeKernelProfile::LightTaeHv15V1
                | VaeKernelProfile::TaeHvHunyuanV1
                | VaeKernelProfile::LightTaeWan21V1
        ),
        "sim.ldm.lightricks.vae.audio_vae.AudioVAE.v1" => {
            matches!(profile, VaeKernelProfile::LtxAudioV1)
        }
        "sim.ldm.audio.vae_sa3.SA3AudioVAE.v1" => matches!(
            profile,
            VaeKernelProfile::StableAudio3DeepV1 | VaeKernelProfile::StableAudio3ShallowV1
        ),
        "sim.ldm.triposplat.vae.OctreeGaussianDecoder.v1" => {
            matches!(profile, VaeKernelProfile::TripoSplatV1)
        }
        _ => false,
    };
    if !matches {
        return Err(VaeArchitectureError::ArchitectureProfileMismatch {
            architecture: architecture.as_str().to_owned(),
            profile: format!("{profile:?}"),
        });
    }
    let configuration_sha256 = match profile {
        VaeKernelProfile::LtxVideoV0 {
            configuration_sha256,
        }
        | VaeKernelProfile::LtxVideoV1 {
            configuration_sha256,
        }
        | VaeKernelProfile::LtxVideoV2 {
            configuration_sha256,
        } => configuration_sha256.as_deref(),
        _ => None,
    };
    if let Some(digest) = configuration_sha256 {
        validate_sha256(digest)?;
    }
    Ok(())
}

pub(crate) fn validate_vae_identity_target(
    architecture: &VaeArchitectureIdentity,
    profile: &VaeKernelProfile,
    family_identity: &ModelFamilyIdentity,
    latent_identity: &LatentFormatIdentity,
    dtype: DType,
    device: DeviceId,
    boundary: crate::VaeBoundaryKind,
) -> Result<(), VaeArchitectureError> {
    validate_architecture_profile_pair(architecture, profile)?;
    if profile.is_conformance_only() {
        return Ok(());
    }
    let (family_registry, latent_registry) = VaeArchitectureRegistry::canonical_targets()?;
    let family = family_registry.definition(family_identity).ok_or_else(|| {
        VaeArchitectureError::UnknownModelFamily(family_identity.identifier().to_owned())
    })?;
    let latent = latent_registry.get(latent_identity).ok_or_else(|| {
        VaeArchitectureError::UnknownLatentFormat(latent_identity.identifier().to_owned())
    })?;
    if family.latent_feature_id != latent.feature_id
        || family.latent_identifier != latent.identifier
    {
        return Err(VaeArchitectureError::FamilyLatentMismatch {
            family: family.identifier.to_owned(),
            expected: format!("{}:{}", family.latent_feature_id, family.latent_identifier),
            actual: format!("{}:{}", latent.feature_id, latent.identifier),
        });
    }
    let contract = profile.contract();
    if boundary != contract.boundary {
        return Err(VaeArchitectureError::ProfileBoundaryMismatch {
            architecture: architecture.as_str().to_owned(),
            expected: contract.boundary,
            actual: boundary,
        });
    }
    match contract.canonical_compatibility {
        VaeCanonicalCompatibility::Exact(allowed) if allowed.contains(&latent.identifier) => {}
        VaeCanonicalCompatibility::Exact(_) => {
            return Err(VaeArchitectureError::ProfileLatentMismatch {
                architecture: architecture.as_str().to_owned(),
                latent: latent.identifier.to_owned(),
            });
        }
        VaeCanonicalCompatibility::Unavailable(reason) => {
            return Err(VaeArchitectureError::CanonicalTargetUnavailable {
                architecture: architecture.as_str().to_owned(),
                reason,
            });
        }
    }
    if latent.dimensions != contract.latent_dimensions
        || contract
            .target_latent_channels
            .is_some_and(|channels| channels != latent.channels)
    {
        return Err(VaeArchitectureError::ProfileLatentMismatch {
            architecture: architecture.as_str().to_owned(),
            latent: latent.identifier.to_owned(),
        });
    }
    if !family.supported_dtypes.contains(&dtype) {
        return Err(VaeArchitectureError::UnsupportedTargetDType {
            family: family.identifier.to_owned(),
            dtype,
        });
    }
    if !contract.supported_dtypes.contains(&dtype) {
        return Err(VaeArchitectureError::UnsupportedProfileDType {
            architecture: architecture.as_str().to_owned(),
            dtype,
        });
    }
    let device_kind = device.kind();
    if !family.supported_devices.contains(&device_kind) {
        return Err(VaeArchitectureError::UnsupportedTargetDevice {
            family: family.identifier.to_owned(),
            device: device_kind,
        });
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), VaeArchitectureError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(VaeArchitectureError::Registry(
            "invalid pinned SHA-256 digest".to_owned(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum VaeArchitectureError {
    #[error("invalid VAE architecture registry: {0}")]
    Registry(String),
    #[error("VAE selector found no source architecture; fail-closed row {unbound_row}")]
    NoMatch { unbound_row: &'static str },
    #[error("VAE selector is ambiguous across source rows {rows:?}")]
    Ambiguous { rows: Vec<&'static str> },
    #[error("VAE selector row {row} is partial: {detail}")]
    Partial { row: &'static str, detail: String },
    #[error("VAE metadata {key} is malformed: {detail}")]
    MalformedMetadata { key: &'static str, detail: String },
    #[error("VAE configuration is malformed: {0}")]
    MalformedConfiguration(String),
    #[error("VAE configuration uses {actual} bytes, exceeding limit {maximum}")]
    ConfigurationLimit { actual: usize, maximum: usize },
    #[error("Diffusers VAE tensor {tensor} is malformed: {detail}")]
    MalformedDiffusersTensor { tensor: String, detail: String },
    #[error("Diffusers VAE conversion maps multiple tensors to {target}")]
    DiffusersKeyCollision { target: String },
    #[error("unknown canonical model family {0}")]
    UnknownModelFamily(String),
    #[error("unknown canonical latent format {0}")]
    UnknownLatentFormat(String),
    #[error("model family {family} requires latent {expected}, got {actual}")]
    FamilyLatentMismatch {
        family: String,
        expected: String,
        actual: String,
    },
    #[error("VAE architecture {architecture} is incompatible with latent format {latent}")]
    ProfileLatentMismatch {
        architecture: String,
        latent: String,
    },
    #[error("VAE architecture {architecture} requires boundary {expected:?}, got {actual:?}")]
    ProfileBoundaryMismatch {
        architecture: String,
        expected: crate::VaeBoundaryKind,
        actual: crate::VaeBoundaryKind,
    },
    #[error("VAE architecture {architecture} has no canonical target: {reason}")]
    CanonicalTargetUnavailable {
        architecture: String,
        reason: &'static str,
    },
    #[error(
        "VAE architecture {architecture} has no generated family for canonical latents {allowed:?}"
    )]
    CanonicalFamilyUnavailable {
        architecture: String,
        allowed: Vec<&'static str>,
    },
    #[error("model family {family} does not support VAE target dtype {dtype:?}")]
    UnsupportedTargetDType { family: String, dtype: DType },
    #[error("VAE architecture {architecture} does not support target dtype {dtype:?}")]
    UnsupportedProfileDType { architecture: String, dtype: DType },
    #[error("model family {family} does not support VAE target device {device:?}")]
    UnsupportedTargetDevice { family: String, device: DeviceKind },
    #[error("VAE architecture {architecture} does not support target device {device:?}")]
    UnsupportedProfileDevice {
        architecture: String,
        device: DeviceKind,
    },
    #[error("native VAE architecture builder is unavailable for {architecture}")]
    ArchitectureUnavailable { architecture: String },
    #[error("VAE architecture {architecture} is incompatible with profile {profile}")]
    ArchitectureProfileMismatch {
        architecture: String,
        profile: String,
    },
    #[error("canonical model-family registry failed: {0}")]
    FamilyRegistry(String),
    #[error("canonical latent-format registry failed: {0}")]
    LatentRegistry(String),
    #[error(transparent)]
    ModelStore(#[from] ModelStoreError),
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selection(
        profile: VaeKernelProfile,
        loader_configuration: VaeLoaderConfiguration,
    ) -> Result<VaeArchitectureSelection, VaeArchitectureError> {
        let architecture = match profile {
            VaeKernelProfile::TemporalAutoencodingEngineV1 => {
                "sim.ldm.models.autoencoder.AutoencodingEngine.temporal.v1"
            }
            VaeKernelProfile::TaesdV1 => "sim.taesd.TAESD.v1",
            VaeKernelProfile::StableCascadeStageAV1 => "sim.ldm.cascade.stage_a.StageA.v1",
            VaeKernelProfile::StableCascadeStageCEncoderV1 => {
                "sim.ldm.cascade.stage_c.StageCEncoder.v1"
            }
            VaeKernelProfile::StableCascadeStageCPreviewerV1 => {
                "sim.ldm.cascade.stage_c.StageCPreviewer.v1"
            }
            VaeKernelProfile::StableCascadeStageCCombinedV1 => {
                "sim.ldm.cascade.stage_c.StageCCombined.v1"
            }
            VaeKernelProfile::HunyuanImageV1 => {
                "sim.ldm.hunyuan_video.vae.AutoencodingEngine.image.v1"
            }
            VaeKernelProfile::HunyuanImageRefinerV1 => {
                "sim.ldm.hunyuan_video.vae_refiner.AutoencodingEngine.image.v1"
            }
            VaeKernelProfile::AutoencoderKlV1
            | VaeKernelProfile::AutoencoderKlX4V1
            | VaeKernelProfile::AutoencoderKlBatchNormV1
            | VaeKernelProfile::ExplicitAutoencoderKlV1
            | VaeKernelProfile::AutoencodingEngineV1
            | VaeKernelProfile::AutoencodingEngineX4V1
            | VaeKernelProfile::AutoencodingEngineBatchNormV1 => {
                "sim.ldm.models.autoencoder.AutoencoderKL.v1"
            }
            VaeKernelProfile::PixelSpaceV1 => "sim.pixel_space_convert.PixelspaceConversionVAE.v1",
            _ => "zed.vae.boundary.v1",
        };
        let contract = profile.contract();
        let boundary = match contract.boundary {
            crate::VaeBoundaryKind::Image => VaeBoundaryDomain::Image,
            crate::VaeBoundaryKind::Video => VaeBoundaryDomain::Video,
            crate::VaeBoundaryKind::Audio => VaeBoundaryDomain::Audio,
            crate::VaeBoundaryKind::StructuredOutput => VaeBoundaryDomain::Structured,
        };
        selected_with(
            architecture,
            profile,
            Vec::new(),
            contract.target_latent_channels,
            contract.target_latent_channels,
            contract.latent_dimensions,
            boundary,
            contract.canonical_compatibility,
            loader_configuration,
            contract.supported_dtypes,
        )
    }

    #[test]
    fn registered_native_adapters_are_constructor_available() -> Result<(), VaeArchitectureError> {
        let automatic_profiles = [
            VaeKernelProfile::TemporalAutoencodingEngineV1,
            VaeKernelProfile::StableCascadeStageAV1,
            VaeKernelProfile::StableCascadeStageCEncoderV1,
            VaeKernelProfile::StableCascadeStageCPreviewerV1,
            VaeKernelProfile::StableCascadeStageCCombinedV1,
            VaeKernelProfile::HunyuanImageV1,
            VaeKernelProfile::PixelSpaceV1,
        ];
        for profile in automatic_profiles {
            selection(profile, VaeLoaderConfiguration::Automatic)?
                .ensure_native_builder_available()?;
        }

        selection(
            VaeKernelProfile::TaesdV1,
            VaeLoaderConfiguration::Taesd {
                latent_channels: 4,
                metadata_override: false,
            },
        )?
        .ensure_native_builder_available()?;

        for profile in [
            VaeKernelProfile::TaeHvWan22V1,
            VaeKernelProfile::TaeHvLtx2V1,
            VaeKernelProfile::LightTaeHv15V1,
            VaeKernelProfile::TaeHvHunyuanV1,
            VaeKernelProfile::LightTaeWan21V1,
            VaeKernelProfile::HunyuanImageRefinerV1,
            VaeKernelProfile::HunyuanVideoRefinerV1,
            VaeKernelProfile::Causal3dV1,
            VaeKernelProfile::CogVideoXV1,
            VaeKernelProfile::CosmosV1,
            VaeKernelProfile::MochiV1,
            VaeKernelProfile::Wan21V1,
            VaeKernelProfile::Wan22V1,
        ] {
            selection(profile, VaeLoaderConfiguration::Automatic)?
                .ensure_native_builder_available()?;
        }
        for profile in [
            VaeKernelProfile::LtxVideoV0 {
                configuration_sha256: None,
            },
            VaeKernelProfile::LtxVideoV1 {
                configuration_sha256: None,
            },
            VaeKernelProfile::LtxVideoV2 {
                configuration_sha256: None,
            },
        ] {
            selection(
                profile,
                VaeLoaderConfiguration::LtxVideo {
                    configuration_sha256: None,
                    configuration_json: None,
                },
            )?
            .ensure_native_builder_available()?;
        }
        for (profile, x4, batch_norm_latent) in [
            (VaeKernelProfile::AutoencoderKlV1, false, false),
            (VaeKernelProfile::AutoencoderKlX4V1, true, false),
            (VaeKernelProfile::AutoencoderKlBatchNormV1, false, true),
            (VaeKernelProfile::AutoencodingEngineV1, false, false),
            (VaeKernelProfile::AutoencodingEngineX4V1, true, false),
            (VaeKernelProfile::AutoencodingEngineBatchNormV1, false, true),
        ] {
            selection(
                profile,
                VaeLoaderConfiguration::DefaultKl {
                    x4,
                    legacy_prefix_rewrite: false,
                    batch_norm_latent,
                    asymmetric_decoder_channels: None,
                    embed_dim: Some(4),
                },
            )?
            .ensure_native_builder_available()?;
        }
        let params = serde_json::json!({
            "ddconfig": {
                "ch": 32,
                "ch_mult": [1, 2],
                "double_z": true,
                "in_channels": 3,
                "num_res_blocks": 1,
                "out_ch": 3,
                "resolution": 8,
                "attn_resolutions": [],
                "resamp_with_conv": true,
                "z_channels": 4
            },
            "embed_dim": 4
        });
        let params_json = canonical_json(&params)?;
        selection(
            VaeKernelProfile::ExplicitAutoencoderKlV1,
            VaeLoaderConfiguration::ExplicitAutoencoderKl {
                params_sha256: format!("{:x}", Sha256::digest(params_json.as_bytes())),
                params_json,
            },
        )?
        .ensure_native_builder_available()?;

        for profile in [
            VaeKernelProfile::AudioOobleck44KhzV1,
            VaeKernelProfile::AudioOobleck48KhzV1,
            VaeKernelProfile::MusicDcaeV1,
            VaeKernelProfile::MmAudio16KhzV1,
            VaeKernelProfile::StableAudio3DeepV1,
            VaeKernelProfile::StableAudio3ShallowV1,
        ] {
            selection(profile, VaeLoaderConfiguration::Automatic)?
                .ensure_native_builder_available()?;
        }
        let empty_configuration = "{}".to_owned();
        let empty_configuration_sha256 =
            format!("{:x}", Sha256::digest(empty_configuration.as_bytes()));
        selection(
            VaeKernelProfile::LtxAudioV1,
            VaeLoaderConfiguration::LtxAudio {
                autoencoder_sha256: empty_configuration_sha256.clone(),
                autoencoder_json: empty_configuration.clone(),
                vocoder_sha256: empty_configuration_sha256,
                vocoder_json: empty_configuration,
                latent_channels: 8,
                input_sample_rate: 16_000,
                output_sample_rate: 16_000,
            },
        )?
        .ensure_native_builder_available()?;

        for profile in [
            VaeKernelProfile::HunyuanShapeV1,
            VaeKernelProfile::TripoSplatV1,
        ] {
            selection(profile, VaeLoaderConfiguration::Automatic)?
                .ensure_native_builder_available()?;
        }

        let unavailable = selection(
            VaeKernelProfile::BlockAverageNearestV1,
            VaeLoaderConfiguration::Automatic,
        )?;
        assert!(matches!(
            unavailable.ensure_native_builder_available(),
            Err(VaeArchitectureError::ArchitectureUnavailable { .. })
        ));
        Ok(())
    }
}
