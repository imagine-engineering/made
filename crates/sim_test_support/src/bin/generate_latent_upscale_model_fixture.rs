use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

const MANIFEST: &str = include_str!(
    "../../fixtures/models/latent-upscale-model-resource-foundation/manifest.canonical.json"
);

fn main() -> Result<()> {
    let check = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(argument) => bail!("unsupported argument {argument}"),
    };
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("workspace root is unavailable")?;
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/models/latent-upscale-model-resource-foundation/manifest.json");
    let mut document: serde_json::Value = serde_json::from_str(MANIFEST)?;
    derive_source_oracles(&mut document)?;
    document["oracle_generator_sha256"] = Value::String(format!(
        "{:x}",
        Sha256::digest(fs::read(workspace.join(file!()))?)
    ));
    let canonical = serde_json::to_string_pretty(&document)? + "\n";
    for source in document
        .get("sources")
        .and_then(serde_json::Value::as_array)
        .context("fixture sources are missing")?
    {
        let path = source
            .get("path")
            .and_then(serde_json::Value::as_str)
            .context("fixture source path is missing")?;
        let expected = source
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .context("fixture source digest is missing")?;
        let actual = format!("{:x}", Sha256::digest(fs::read(workspace.join(path))?));
        if actual != expected {
            bail!("pinned source digest changed for {path}");
        }
    }
    if check {
        if fs::read_to_string(&fixture)? != canonical {
            bail!("latent upscale fixture is stale");
        }
    } else {
        fs::write(fixture, canonical)?;
    }
    Ok(())
}

fn derive_source_oracles(document: &mut Value) -> Result<()> {
    document["oracle_domain"] =
        Value::String("zed.sim.latent-upscale-independent-source-equations.v1".to_owned());
    let cases = document
        .get_mut("cases")
        .and_then(Value::as_array_mut)
        .context("fixture cases are missing")?;
    for case in &mut *cases {
        let identifier = case
            .get("id")
            .and_then(Value::as_str)
            .context("fixture case id is missing")?;
        match identifier {
            "720-integrated-residual-order" => {
                let expected = f32_array(case, "input")?
                    .into_iter()
                    .map(|value| value + silu(silu(value)))
                    .collect::<Vec<_>>();
                set_f32_oracle(case, "expected", &expected);
            }
            "1080-repeat-rms-shortcut-order" => {
                let input = f32_array(case, "input")?;
                let convolution = f32_array(case, "conv_in_weights")?;
                let output_weights = f32_array(case, "output_weights")?;
                let branches = nested_f32_arrays(case, "residual_branch_weights")?;
                let repeat = usize_field(case, "repeat_count")?;
                let block_count = usize_field(case, "residual_blocks_per_stage")?;
                if input.len() != 1
                    || convolution.len() != repeat
                    || output_weights.len() != repeat
                    || branches.len() != block_count
                    || branches.iter().any(|branch| branch.len() != repeat)
                {
                    bail!("1080 oracle configuration is inconsistent");
                }
                let repeated = vec![input[0]; repeat];
                let mut hidden = convolution
                    .iter()
                    .zip(&repeated)
                    .map(|(weight, repeated)| weight * input[0] + repeated)
                    .collect::<Vec<_>>();
                for branch in branches {
                    let first = hunyuan_rms(&hidden)
                        .into_iter()
                        .map(silu)
                        .collect::<Vec<_>>();
                    let second = hunyuan_rms(&first)
                        .into_iter()
                        .map(silu)
                        .collect::<Vec<_>>();
                    for ((value, branch), activated) in hidden.iter_mut().zip(branch).zip(second) {
                        *value += branch * activated;
                    }
                }
                let expected = hunyuan_rms(&hidden)
                    .into_iter()
                    .map(silu)
                    .zip(output_weights)
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>();
                set_f32_oracle(case, "expected", &[expected]);
            }
            "pixel-shuffle-dimension-one" => {
                let channels = nested_f32_arrays(case, "input_channels")?;
                if channels.len() != 2 || channels.iter().any(|channel| channel.len() != 2) {
                    bail!("dimension-one shuffle oracle requires two channels and two frames");
                }
                let mut expected = Vec::new();
                for frame in 0..2 {
                    for channel in &channels {
                        expected.push(channel[frame]);
                    }
                }
                set_f32_oracle(case, "expected_frames", &expected);
            }
            "pixel-shuffle-dimension-two" => {
                let channels = f32_array(case, "input_channels")?;
                if channels.len() != 4 {
                    bail!("dimension-two shuffle oracle requires four channels");
                }
                case["expected"] = json!([[channels[0], channels[1]], [channels[2], channels[3]]]);
                case["expected_bits"] = json!([
                    [channels[0].to_bits(), channels[1].to_bits()],
                    [channels[2].to_bits(), channels[3].to_bits()]
                ]);
            }
            "pixel-shuffle-dimension-three" => {
                let channels = f32_array(case, "input_channels")?;
                if channels.len() != 8 {
                    bail!("dimension-three shuffle oracle requires eight channels");
                }
                case["expected_frames"] = json!([
                    [[channels[0], channels[1]], [channels[2], channels[3]]],
                    [[channels[4], channels[5]], [channels[6], channels[7]]]
                ]);
                case["expected_bits"] = json!([
                    [
                        [channels[0].to_bits(), channels[1].to_bits()],
                        [channels[2].to_bits(), channels[3].to_bits()]
                    ],
                    [
                        [channels[4].to_bits(), channels[5].to_bits()],
                        [channels[6].to_bits(), channels[7].to_bits()]
                    ]
                ]);
            }
            "rational-blur-center-delta" => {
                let kernel = [1_u32, 4, 6, 4, 1];
                let sampled = [kernel[0], kernel[2], kernel[4]];
                case["expected_numerator"] = json!([
                    [
                        sampled[0] * sampled[0],
                        sampled[0] * sampled[1],
                        sampled[0] * sampled[2]
                    ],
                    [
                        sampled[1] * sampled[0],
                        sampled[1] * sampled[1],
                        sampled[1] * sampled[2]
                    ],
                    [
                        sampled[2] * sampled[0],
                        sampled[2] * sampled[1],
                        sampled[2] * sampled[2]
                    ]
                ]);
                case["normalization_denominator"] = json!(256);
            }
            "nearest-exact-half-coordinate" => {
                let input = f32_array(case, "input")?;
                let target = usize_field(case, "target_length")?;
                if input.is_empty() || target == 0 {
                    bail!("nearest-exact oracle dimensions are empty");
                }
                let expected = (0..target)
                    .map(|index| {
                        let source = ((index as f64 + 0.5) * input.len() as f64 / target as f64)
                            .floor() as usize;
                        input[source.min(input.len() - 1)]
                    })
                    .collect::<Vec<_>>();
                set_f32_oracle(case, "expected", &expected);
            }
            "bislerp-edge-cases" => {
                let pairs = case
                    .get("pairs")
                    .and_then(Value::as_array)
                    .context("bislerp oracle pairs are missing")?;
                let expected = pairs
                    .iter()
                    .map(|pair| {
                        source_slerp(
                            &f32_array(pair, "left")?,
                            &f32_array(pair, "right")?,
                            f32_field(pair, "ratio")?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                case["expected"] = json!(expected);
                case["expected_bits"] = json!(
                    expected
                        .iter()
                        .map(|values| values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                );
            }
            "ltx-vae-statistics-order" => {
                let input = f32_array(case, "input")?;
                let means = f32_array(case, "mean")?;
                let standard_deviations = f32_array(case, "standard_deviation")?;
                let model_raw = f32_array(case, "model_raw")?;
                if input.len() != means.len()
                    || input.len() != standard_deviations.len()
                    || input.len() != model_raw.len()
                {
                    bail!("VAE oracle vectors have inconsistent lengths");
                }
                let unnormalized = input
                    .iter()
                    .zip(&standard_deviations)
                    .zip(&means)
                    .map(|((value, standard_deviation), mean)| value * standard_deviation + mean)
                    .collect::<Vec<_>>();
                let normalized = model_raw
                    .iter()
                    .zip(&means)
                    .zip(&standard_deviations)
                    .map(|((value, mean), standard_deviation)| (value - mean) / standard_deviation)
                    .collect::<Vec<_>>();
                set_f32_oracle(case, "expected_unnormalized", &unnormalized);
                set_f32_oracle(case, "expected_normalized", &normalized);
            }
            other => bail!("unsupported latent-upscale oracle case {other}"),
        }
    }
    let oracle_bytes = serde_json::to_vec(cases)?;
    document["oracle_outputs_sha256"] =
        Value::String(format!("{:x}", Sha256::digest(oracle_bytes)));
    Ok(())
}

fn f32_array(case: &Value, field: &str) -> Result<Vec<f32>> {
    case.get(field)
        .and_then(Value::as_array)
        .context("oracle vector is missing")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .context("oracle vector contains a non-number")
        })
        .collect()
}

fn set_f32_oracle(case: &mut Value, field: &str, values: &[f32]) {
    case[field] = json!(values);
    case[format!("{field}_bits")] = json!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn hunyuan_rms(values: &[f32]) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1.0e-12);
    let scale = (values.len() as f32).sqrt();
    values.iter().map(|value| value / norm * scale).collect()
}

fn nested_f32_arrays(case: &Value, field: &str) -> Result<Vec<Vec<f32>>> {
    case.get(field)
        .and_then(Value::as_array)
        .context("nested oracle vector is missing")?
        .iter()
        .map(|value| f32_array(&json!({"value": value}), "value"))
        .collect()
}

fn usize_field(case: &Value, field: &str) -> Result<usize> {
    usize::try_from(
        case.get(field)
            .and_then(Value::as_u64)
            .context("oracle integer field is missing")?,
    )
    .context("oracle integer does not fit usize")
}

fn f32_field(case: &Value, field: &str) -> Result<f32> {
    case.get(field)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .context("oracle floating-point field is missing")
}

fn source_slerp(left: &[f32], right: &[f32], ratio: f32) -> Result<Vec<f32>> {
    if left.len() != right.len() || left.is_empty() {
        bail!("bislerp oracle pair is invalid");
    }
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    let normalized_left = left
        .iter()
        .map(|value| {
            if left_norm == 0.0 {
                0.0
            } else {
                value / left_norm
            }
        })
        .collect::<Vec<_>>();
    let normalized_right = right
        .iter()
        .map(|value| {
            if right_norm == 0.0 {
                0.0
            } else {
                value / right_norm
            }
        })
        .collect::<Vec<_>>();
    let dot = normalized_left
        .iter()
        .zip(&normalized_right)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    if dot > 1.0 - 1.0e-5 {
        return Ok(left.to_vec());
    }
    if dot < 1.0e-5 - 1.0 {
        return Ok(left
            .iter()
            .zip(right)
            .map(|(left, right)| left * (1.0 - ratio) + right * ratio)
            .collect());
    }
    let omega = dot.acos();
    let sine = omega.sin();
    let length = left_norm * (1.0 - ratio) + right_norm * ratio;
    Ok(normalized_left
        .iter()
        .zip(&normalized_right)
        .map(|(left, right)| {
            ((((1.0 - ratio) * omega).sin() / sine) * left + ((ratio * omega).sin() / sine) * right)
                * length
        })
        .collect())
}
