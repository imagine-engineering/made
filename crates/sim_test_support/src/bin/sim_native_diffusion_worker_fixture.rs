use sim_test_support::NativeDiffusionFixture;
use std::{env, path::PathBuf, sync::Arc};

const DEFAULT_WORKER_MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    let (memory_limit_bytes, model_root) = parse_arguments()?;
    smol::block_on(sim_worker::run_worker_process_with_diffusion_provider(
        memory_limit_bytes,
        Some(Arc::new(NativeDiffusionFixture::at(model_root))),
    ))
}

fn parse_arguments() -> anyhow::Result<(u64, PathBuf)> {
    let mut memory_limit_bytes = DEFAULT_WORKER_MEMORY_LIMIT_BYTES;
    let mut model_root = None;
    let mut backend_seen = false;
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--memory-limit-bytes" {
            let value = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("--memory-limit-bytes requires a value"))?;
            memory_limit_bytes = value
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("memory limit must be UTF-8 decimal bytes"))?
                .parse::<u64>()
                .map_err(|error| anyhow::anyhow!("invalid worker memory limit: {error}"))?;
        } else if argument == "--fixture-model-root" {
            model_root =
                Some(PathBuf::from(arguments.next().ok_or_else(|| {
                    anyhow::anyhow!("--fixture-model-root requires a path")
                })?));
        } else if argument == "--backend" {
            let backend = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("--backend requires a value"))?;
            if backend_seen || backend != "cpu" {
                return Err(anyhow::anyhow!(
                    "native diffusion fixture requires exactly one CPU backend selection"
                ));
            }
            backend_seen = true;
        } else {
            return Err(anyhow::anyhow!("unknown native diffusion worker argument"));
        }
    }
    if !backend_seen {
        return Err(anyhow::anyhow!(
            "native diffusion fixture requires an explicit CPU backend selection"
        ));
    }
    Ok((
        memory_limit_bytes,
        model_root.ok_or_else(|| anyhow::anyhow!("--fixture-model-root is required"))?,
    ))
}
