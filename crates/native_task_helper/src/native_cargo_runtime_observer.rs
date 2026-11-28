#[cfg(target_os = "macos")]
fn main() {
    native_task_helper::macos_task_completion::run_native_cargo_runtime_helper();
}

#[cfg(target_os = "linux")]
fn main() {
    native_task_helper::linux_cargo_runtime::run_linux_cargo_runtime_helper();
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("Native Cargo runtime observation is unavailable on this host");
    std::process::exit(2);
}
