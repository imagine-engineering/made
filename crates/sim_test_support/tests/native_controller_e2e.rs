#[path = "support/accelerator_selection.rs"]
mod accelerator_selection;
#[path = "support/native_controller.rs"]
mod native_controller;

#[test]
fn native_controller_drives_packaged_worker_commands_and_typed_outputs()
-> Result<(), Box<dyn std::error::Error>> {
    let cases = native_controller::run_native_controller_e2e()?;
    assert!(cases.values().all(|passed| *passed));
    let accelerator_cases = accelerator_selection::accelerator_selection_contract_cases();
    assert!(
        accelerator_cases.values().all(|passed| *passed),
        "accelerator selection contract cases failed: {accelerator_cases:#?}"
    );
    Ok(())
}
