use std::{
    env,
    io::{self, Write as _},
    process::{Command, ExitStatus},
};

use buzz_compat::{
    BUZZ_CLI_VERSION, EndpointCompatibility, ForwardRequest, ForwardRunner, SHIM_PROTOCOL_VERSION,
    ShimEnvironment, ShimExecution, Version, execute,
};

struct ProcessRunner;

impl ForwardRunner for ProcessRunner {
    #[allow(
        clippy::disallowed_methods,
        reason = "the standalone synchronous shim must inherit and wait for the forwarded CLI streams"
    )]
    fn run(&self, request: ForwardRequest) -> io::Result<ShimExecution> {
        let status = Command::new(request.program)
            .args(request.args)
            .envs(request.environment)
            .status()?;
        Ok(ShimExecution {
            stdout: Vec::new(),
            stderr: Vec::new(),
            exit_code: status_code(status),
        })
    }
}

fn status_code(status: ExitStatus) -> i32 {
    status.code().unwrap_or(4)
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let environment = ShimEnvironment::from_variables(env::vars());
    let compatibility = match compatibility(&environment) {
        Ok(compatibility) => compatibility,
        Err(error) => {
            let execution = ShimExecution {
                stdout: Vec::new(),
                stderr: format!(
                    "{{\"error\":\"user_error\",\"message\":\"{}\",\"retryable\":false}}\n",
                    error
                )
                .into_bytes(),
                exit_code: 1,
            };
            finish(execution);
        }
    };
    let program = environment
        .get("ZED_COLLABORATION_CLI")
        .unwrap_or("zed")
        .to_owned();
    finish(execute(
        &ProcessRunner,
        program,
        &args,
        &environment,
        compatibility,
    ));
}

fn compatibility(environment: &ShimEnvironment) -> Result<EndpointCompatibility, &'static str> {
    let protocol_version = environment
        .get("ZED_COLLABORATION_PROTOCOL_VERSION")
        .map(str::parse)
        .transpose()
        .map_err(|_| "invalid ZED_COLLABORATION_PROTOCOL_VERSION")?
        .unwrap_or(SHIM_PROTOCOL_VERSION);
    let minimum_buzz_cli_version = environment
        .get("ZED_COLLABORATION_MIN_BUZZ_CLI_VERSION")
        .map(Version::parse)
        .transpose()
        .map_err(|_| "invalid ZED_COLLABORATION_MIN_BUZZ_CLI_VERSION")?
        .unwrap_or(BUZZ_CLI_VERSION);
    Ok(EndpointCompatibility {
        protocol_version,
        minimum_buzz_cli_version,
    })
}

fn finish(execution: ShimExecution) -> ! {
    if let Err(error) = io::stdout().write_all(&execution.stdout) {
        write_failure("stdout", &error);
    }
    if let Err(error) = io::stderr().write_all(&execution.stderr) {
        write_failure("stderr", &error);
    }
    std::process::exit(execution.exit_code)
}

fn write_failure(stream: &str, error: &io::Error) -> ! {
    eprintln!("buzz compatibility shim failed to write {stream}: {error}");
    std::process::exit(4)
}
