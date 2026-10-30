#[cfg(target_os = "macos")]
fn main() {
    native_task_helper::macos_task_completion::run_native_cargo_runtime_helper();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Native Cargo runtime observation is unavailable on this host");
    std::process::exit(2);
}
