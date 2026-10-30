use std::{env, fs, path::PathBuf, time::Duration};

use sim_runtime::PluginAuthorizationVerifier;

const DEFAULT_WORKER_MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

fn main() -> anyhow::Result<()> {
    let mut memory_limit_bytes = DEFAULT_WORKER_MEMORY_LIMIT_BYTES;
    let mut exit_after_milliseconds = None;
    let mut exit_marker = None;
    let mut plugin_authorization_verifier = None;
    let mut backend_seen = false;
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--memory-limit-bytes" {
            let value = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("memory limit is missing"))?;
            memory_limit_bytes = value
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("memory limit is not UTF-8"))?
                .parse::<u64>()?;
        } else if argument == "--exit-after-ms-once" {
            let value = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("exit delay is missing"))?;
            exit_after_milliseconds = Some(
                value
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("exit delay is not UTF-8"))?
                    .parse::<u64>()?,
            );
        } else if argument == "--exit-marker" {
            exit_marker = Some(PathBuf::from(
                arguments
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("exit marker is missing"))?,
            ));
        } else if argument == "--plugin-authorization-verification-key" {
            let value = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("authorization verifier is missing"))?;
            let value = value
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("authorization verifier is not UTF-8"))?;
            if plugin_authorization_verifier
                .replace(PluginAuthorizationVerifier::from_token(value)?)
                .is_some()
            {
                return Err(anyhow::anyhow!(
                    "authorization verifier was provided more than once"
                ));
            }
        } else if argument == "--backend" {
            let backend = arguments
                .next()
                .ok_or_else(|| anyhow::anyhow!("backend selection is missing"))?;
            if backend_seen || backend != "cpu" {
                return Err(anyhow::anyhow!(
                    "plugin fixture requires exactly one CPU backend selection"
                ));
            }
            backend_seen = true;
        } else {
            return Err(anyhow::anyhow!("unknown worker argument"));
        }
    }
    if !backend_seen {
        return Err(anyhow::anyhow!(
            "plugin fixture requires an explicit CPU backend selection"
        ));
    }
    match (exit_after_milliseconds, exit_marker) {
        (Some(delay), Some(marker)) if !marker.try_exists()? => {
            fs::write(&marker, b"private worker loss injected\n")?;
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(delay));
                std::process::exit(91);
            });
        }
        (None, None) | (Some(_), Some(_)) => {}
        _ => {
            return Err(anyhow::anyhow!(
                "exit delay and marker must be provided together"
            ));
        }
    }
    smol::block_on(sim_worker::run_worker_process_with_authorization_verifier(
        memory_limit_bytes,
        plugin_authorization_verifier,
    ))
}
