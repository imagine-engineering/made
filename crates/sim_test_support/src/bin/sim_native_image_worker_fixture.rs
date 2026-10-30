use std::env;

const DEFAULT_WORKER_MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    let memory_limit_bytes = parse_memory_limit()?;
    smol::block_on(sim_worker::run_worker_process(memory_limit_bytes))
}

fn parse_memory_limit() -> anyhow::Result<u64> {
    let mut memory_limit_bytes = DEFAULT_WORKER_MEMORY_LIMIT_BYTES;
    let mut backend_seen = false;
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--backend" {
            let backend = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("--backend requires a value"))?;
            if backend_seen || backend != "cpu" {
                return Err(anyhow::anyhow!(
                    "native image fixture requires exactly one CPU backend selection"
                ));
            }
            backend_seen = true;
            continue;
        }
        if argument != "--memory-limit-bytes" {
            return Err(anyhow::anyhow!("unknown native image worker argument"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| anyhow::anyhow!("--memory-limit-bytes requires a value"))?;
        let value = value
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("memory limit must be UTF-8 decimal bytes"))?;
        memory_limit_bytes = value
            .parse::<u64>()
            .map_err(|error| anyhow::anyhow!("invalid worker memory limit: {error}"))?;
    }
    if !backend_seen {
        return Err(anyhow::anyhow!(
            "native image fixture requires an explicit CPU backend selection"
        ));
    }
    Ok(memory_limit_bytes)
}
