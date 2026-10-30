use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sim_media::{PngLimits, encode_png_frame};
use sim_model::generated_native_diffusion::{
    SD15_FEATURE_ID, SD15_TINY_FIXTURE_ID, Sd15DetectorProjection, empty_sd15_latent,
    encode_sd15_prompt, sd15_tiny_weight_manifest,
};
use sim_runtime::Sd15GuidanceAdapter;
use sim_sampler::NoiseRequest;
use sim_sampler::generated_native_diffusion::{
    checked_native_diffusion_plan, normal_noise, normal_sigmas, sample_euler, scale_initial_noise,
    scale_model_input, sd15_interpret_prediction,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DeviceId, ExecutionContext, StreamId,
    generated_native_diffusion::tensor_to_f32,
};
use sim_test_support::NativeDiffusionFixture;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

const SEED: u64 = 0x0123_4567_89ab_cdef;
const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const FIXTURE_PROMPT_ID: &str = "53494d00-0000-0000-0000-000000003702";
const FIXTURE_KSAMPLER_NODE_ID: &str = "5";

fn main() -> Result<()> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("workspace root is unavailable")?;
    let mut arguments = std::env::args_os().skip(1);
    let root = match arguments.next() {
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models/sd15-tiny-v1"),
        Some(argument) if argument == "--output-root" => {
            let root = PathBuf::from(
                arguments
                    .next()
                    .context("--output-root requires a directory")?,
            );
            if arguments.next().is_some() || root.as_os_str().is_empty() {
                bail!("usage: generate_native_diffusion_fixture [--output-root directory]");
            }
            root
        }
        Some(_) => bail!("usage: generate_native_diffusion_fixture [--output-root directory]"),
    };
    fs::create_dir_all(&root)?;
    copy_tokenizer(workspace, &root)?;
    write_detector(&root)?;
    let specs = sd15_tiny_weight_manifest()?;
    write_manifest(&root, &specs)?;
    let model_tensors = specs
        .iter()
        .map(|spec| {
            Ok((
                spec.key.clone(),
                spec.shape.clone(),
                deterministic_values(&spec.key, element_count(&spec.shape)?),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    write_safetensors(&root.join("model.safetensors"), &model_tensors)?;
    write_provenance(workspace, &root)?;

    let cancellation = CancellationToken::default();
    let fixture = NativeDiffusionFixture::at(&root);
    let tokenizer = fixture.tokenizer()?;
    let positive_tokens = encode_sd15_prompt(&tokenizer, "a test", &cancellation)?;
    let negative_tokens = encode_sd15_prompt(&tokenizer, "", &cancellation)?;
    if positive_tokens[..4] != [49_406, 320, 1_628, 49_407]
        || negative_tokens[0] != 49_406
        || negative_tokens[1..].iter().any(|token| *token != 49_407)
    {
        bail!("SD1 tokenizer did not produce the pinned token IDs");
    }
    write_json(
        &root.join("tokens.json"),
        &json!({
            "fixture_id": SD15_TINY_FIXTURE_ID,
            "negative": negative_tokens.as_slice(),
            "positive": positive_tokens.as_slice(),
        }),
    )?;

    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let backend = Arc::new(backend);
    let workspace = workspace_authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let bundle = fixture.load_bundle_with_context(backend.clone(), &context)?;
    let (_, positive) = bundle.encode_text("a test", &context)?;
    let (_, negative) = bundle.encode_text("", &context)?;
    let model = bundle.model();
    {
        let negative_values = tensor_to_f32(&backend, &negative, &context)?;
        let positive_values = tensor_to_f32(&backend, &positive, &context)?;
        write_safetensors(
            &root.join("clip-conditioning.safetensors"),
            &[
                ("negative".to_owned(), vec![1, 77, 32], &negative_values[..]),
                ("positive".to_owned(), vec![1, 77, 32], &positive_values[..]),
            ],
        )?;
    }

    let sigmas = normal_sigmas(&backend, &context, 4, 1.0)?;
    fs::write(
        root.join("normal-sigmas.f64le"),
        sigmas
            .iter()
            .flat_map(|value| f64::from(*value).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let rng = NoiseRequest::native_diffusion(FIXTURE_PROMPT_ID, FIXTURE_KSAMPLER_NODE_ID)?
        .stream(SEED, DeviceId::CPU)?;
    let noise = normal_noise(&backend, &[1, 4, 4, 4], &rng, &context)?;
    fs::write(
        root.join("rng-state-before-noise.bin"),
        serde_json::to_vec(&noise.before)?,
    )?;
    write_single_tensor(
        &root.join("initial-noise.safetensors"),
        "noise",
        &noise.noise,
        &backend,
        &context,
    )?;
    let latent = empty_sd15_latent(&backend, 1, 32, 32, &context)?;
    let initial = scale_initial_noise(&backend, &noise.noise, &latent, sigmas[0], &context)?;
    let plan = checked_native_diffusion_plan("euler", "normal", SEED, 4, 7.0, 1.0)?;
    let mut guidance =
        Sd15GuidanceAdapter::checked(model.as_ref(), &positive, &negative, &context)?;
    let trace = sample_euler(
        &backend,
        initial,
        &sigmas,
        &context,
        |latent, sigma, _step| {
            let model_input = scale_model_input(&backend, latent, sigma, &context)
                .map_err(|error| error.to_string())?;
            let prediction = guidance
                .execute(&backend, &model_input, sigma, &plan, &context)
                .map_err(|error| error.to_string())?;
            sd15_interpret_prediction(&backend, prediction.guided(), latent, sigma, &context)
                .map_err(|error| error.to_string())
        },
    )?;
    for (index, denoised) in trace.denoiser_evaluations.iter().enumerate() {
        write_single_tensor(
            &root.join(format!("denoiser-eval-{index:03}.safetensors")),
            "denoised",
            denoised,
            &backend,
            &context,
        )?;
    }
    for (index, latent) in trace.latents.iter().enumerate() {
        write_single_tensor(
            &root.join(format!("latent-step-{index:03}.safetensors")),
            "latent",
            latent,
            &backend,
            &context,
        )?;
    }
    let final_latent = trace
        .latents
        .last()
        .context("Euler trace has no final latent")?;
    let decoded = bundle
        .vae()
        .decode(backend.as_ref(), final_latent, &context)?;
    let decoded_nchw = tensor_to_f32(&backend, &decoded, &context)?;
    fs::write(
        root.join("vae-decoded.f32le"),
        decoded_nchw
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let decoded_bhwc = nchw_to_bhwc(&decoded_nchw, 3, 32, 32)?;
    let png = encode_png_frame(
        &decoded_bhwc,
        1,
        32,
        32,
        3,
        0,
        &BTreeMap::new(),
        PngLimits::default(),
    )?;
    fs::write(root.join("output.png"), png)?;
    write_json(
        &root.join("expected-events.json"),
        &json!({
            "events": [
                "model_loaded", "positive_conditioning_ready", "negative_conditioning_ready",
                "noise_ready", "sampler_step_0", "sampler_step_1", "sampler_step_2",
                "sampler_step_3", "vae_decoded", "image_saved"
            ],
            "fixture_id": SD15_TINY_FIXTURE_ID,
        }),
    )?;
    Ok(())
}

fn copy_tokenizer(workspace: &Path, root: &Path) -> Result<()> {
    let source = workspace.join("projects/comfy/ComfyUI/comfy/sd1_tokenizer");
    let vocabulary = root.join("vocab.json");
    let merges = root.join("merges.txt");
    fs::copy(source.join("vocab.json"), &vocabulary)?;
    fs::copy(source.join("merges.txt"), &merges)?;
    normalize_fixture_permissions(&vocabulary)?;
    normalize_fixture_permissions(&merges)?;
    Ok(())
}

#[cfg(unix)]
fn normalize_fixture_permissions(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o644))?;
    Ok(())
}

#[cfg(not(unix))]
fn normalize_fixture_permissions(path: &Path) -> Result<()> {
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn write_detector(root: &Path) -> Result<()> {
    let source_shapes = BTreeMap::from([
        (
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight".to_owned(),
            vec![49_408, 768],
        ),
        (
            "first_stage_model.quant_conv.weight".to_owned(),
            vec![8, 8, 1, 1],
        ),
        (
            "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
            vec![320, 4, 3, 3],
        ),
        (
            "model.diffusion_model.out.2.weight".to_owned(),
            vec![4, 320, 3, 3],
        ),
        (
            "model.diffusion_model.time_embed.0.weight".to_owned(),
            vec![1280, 320],
        ),
    ]);
    write_json(
        &root.join("sd15-detector-projection.json"),
        &Sd15DetectorProjection {
            feature_id: SD15_FEATURE_ID.to_owned(),
            model_channels: 320,
            context_dim: 768,
            adm_in_channels: None,
            use_linear_in_transformer: false,
            use_temporal_attention: false,
            source_shapes,
        },
    )
}

fn write_manifest(
    root: &Path,
    specs: &[sim_model::generated_native_diffusion::WeightSpec],
) -> Result<()> {
    let weights = specs
        .iter()
        .map(|spec| {
            json!({
                "dtype": "F32", "element_count": element_count(&spec.shape).unwrap_or(0),
                "key": spec.key, "shape": spec.shape,
            })
        })
        .collect::<Vec<_>>();
    write_json(
        &root.join("state-dict-manifest.json"),
        &json!({
            "fixture_id": SD15_TINY_FIXTURE_ID,
            "prefix_map": {
                "cond_stage_model.": "clip_l.",
                "first_stage_model.": "",
                "model.diffusion_model.": ""
            },
            "schema_version": 1,
            "weight_generation": "splitmix64-sha256-key-signed-top24-f32-v1",
            "weights": weights,
        }),
    )
}

fn write_provenance(workspace: &Path, root: &Path) -> Result<()> {
    let paths = [
        "projects/comfy/ComfyUI/comfy/supported_models.py",
        "projects/comfy/ComfyUI/comfy/supported_models_base.py",
        "projects/comfy/ComfyUI/comfy/sd1_clip.py",
        "projects/comfy/ComfyUI/comfy/sd1_clip_config.json",
        "projects/comfy/ComfyUI/comfy/latent_formats.py",
        "projects/comfy/ComfyUI/comfy/sd.py",
        "projects/comfy/ComfyUI/comfy/ldm/modules/diffusionmodules/openaimodel.py",
        "projects/comfy/ComfyUI/comfy/ldm/models/autoencoder.py",
        "projects/comfy/ComfyUI/comfy/samplers.py",
        "projects/comfy/ComfyUI/comfy/k_diffusion/sampling.py",
        "projects/comfy/ComfyUI/nodes.py",
    ];
    let mut sources = BTreeMap::new();
    for path in paths {
        sources.insert(
            path,
            format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?)),
        );
    }
    write_json(
        &root.join("oracle-provenance.json"),
        &json!({
            "fixture_id": SD15_TINY_FIXTURE_ID,
            "oracle_kind": "pinned-comfyui-source-equation-fixture",
            "production_dependency": false,
            "sources": sources,
            "weight_algorithm": "SHA-256 key seed plus SplitMix64 signed top-24 stream",
        }),
    )
}

fn deterministic_values(key: &str, count: usize) -> Vec<f32> {
    let mut digest = Sha256::new();
    digest.update(SD15_TINY_FIXTURE_ID.as_bytes());
    digest.update([0]);
    digest.update(key.as_bytes());
    let digest = digest.finalize();
    let mut seed_bytes = [0_u8; 8];
    seed_bytes.copy_from_slice(&digest[..8]);
    let mut state = u64::from_le_bytes(seed_bytes);
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            let top = (value >> 40) as i32;
            let signed = if top & 0x0080_0000 != 0 {
                top - 0x0100_0000
            } else {
                top
            };
            (signed as f32 / 8_388_608.0) * 0.02
        })
        .collect()
}

fn element_count(shape: &[u64]) -> Result<usize> {
    let count = shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .context("tensor shape overflow")?;
    usize::try_from(count).context("tensor is too large")
}

fn write_single_tensor(
    path: &Path,
    name: &str,
    tensor: &sim_tensor::Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<()> {
    let values = tensor_to_f32(backend, tensor, context)?;
    write_safetensors(
        path,
        &[(
            name.to_owned(),
            tensor.descriptor().shape().to_vec(),
            &values[..],
        )],
    )
}

fn write_safetensors<T>(path: &Path, tensors: &[(String, Vec<u64>, T)]) -> Result<()>
where
    T: AsRef<[f32]>,
{
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for value in values.as_ref() {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut encoded_header = serde_json::to_vec(&header)?;
    while !encoded_header.len().is_multiple_of(8) {
        encoded_header.push(b' ');
    }
    let mut encoded = Vec::with_capacity(8 + encoded_header.len() + data.len());
    encoded.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&encoded_header);
    encoded.extend_from_slice(&data);
    fs::write(path, encoded)?;
    Ok(())
}

fn nchw_to_bhwc(values: &[f32], channels: usize, height: usize, width: usize) -> Result<Vec<f32>> {
    if values.len() != channels * height * width {
        bail!("decoded tensor shape is invalid");
    }
    let mut output = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                output[(y * width + x) * channels + channel] =
                    values[(channel * height + y) * width + x];
            }
        }
    }
    Ok(output)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}
