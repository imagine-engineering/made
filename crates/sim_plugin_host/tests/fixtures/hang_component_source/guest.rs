wit_bindgen::generate!({
    path: "../../../../sim_plugin_sdk/wit",
    world: "sim-plugin",
});

use exports::zed::sim_plugin::plugin::Guest;
use zed::sim_plugin::types;

struct HangComponent;

impl Guest for HangComponent {
    fn manifest() -> types::ManifestProjection {
        spin_forever()
    }

    fn create_node(_node_id: String) -> Result<u64, types::InvocationError> {
        spin_forever()
    }

    fn invoke(_instance: u64) -> Result<(), types::InvocationError> {
        spin_forever()
    }

    fn cancel(_instance: u64, _reason: types::CancelReason) -> Result<(), types::InvocationError> {
        spin_forever()
    }

    fn drop_node(_instance: u64) {
        spin_forever()
    }
}

fn spin_forever<T>() -> T {
    loop {
        core::hint::spin_loop();
    }
}

export!(HangComponent);
