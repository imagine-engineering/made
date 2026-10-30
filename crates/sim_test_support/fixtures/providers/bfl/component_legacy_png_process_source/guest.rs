wit_bindgen::generate!({
    path: "../../../../../sim_plugin_sdk/wit",
    world: "sim-provider-plugin",
});
use exports::zed::sim_plugin::{plugin, provider_binding};
use zed::sim_plugin::{host, types};
include!(env!("SIM_BFL_PNG_GUEST_PROJECTION_RS"));

struct BflLegacyPngProcessProbe;
impl plugin::Guest for BflLegacyPngProcessProbe {
    fn manifest() -> types::ManifestProjection {
        fixture_manifest()
    }
    fn create_node(node_id: String) -> Result<u64, types::InvocationError> {
        if node_id == "FluxProUltraImageNode" {
            Ok(1)
        } else {
            Err(types::InvocationError::PluginFailure("development probe only executes Ultra; other full-namespace descriptors do not certify Source implementations".to_owned()))
        }
    }
    fn invoke(_instance: u64) -> Result<(), types::InvocationError> {
        Err(types::InvocationError::PluginFailure(
            "requires legacy provider invocation".to_owned(),
        ))
    }
    fn cancel(_instance: u64, _reason: types::CancelReason) -> Result<(), types::InvocationError> {
        Ok(())
    }
    fn drop_node(_instance: u64) {}
}
impl provider_binding::Guest for BflLegacyPngProcessProbe {
    fn binding_set() -> types::ProviderBindingSet {
        fixture_binding_set()
    }
    fn invoke_provider(
        class_type: String,
        request: Vec<u8>,
    ) -> Result<types::ProviderInvocationResponse, types::InvocationError> {
        if class_type != "FluxProUltraImageNode"
            || !request.starts_with(b"zed.sim.provider-transport-request\0")
        {
            return Err(types::InvocationError::PluginFailure(
                "unexpected development probe invocation".to_owned(),
            ));
        }
        let mut receipt = host::provider_request(
            "fixture",
            "https://fixture.invalid/v1/generate",
            &request,
            None,
        )?;
        if request
            .windows(b"fixture-invalid-receipt".len())
            .any(|window| window == b"fixture-invalid-receipt")
        {
            let mut wire: serde_json::Value = serde_json::from_slice(&receipt).map_err(|_| {
                types::InvocationError::HostFailure(
                    "broker receipt must retain its public JSON encoding".to_owned(),
                )
            })?;
            let object = wire.as_object_mut().ok_or_else(|| {
                types::InvocationError::HostFailure(
                    "broker receipt wire must be an object".to_owned(),
                )
            })?;
            object.insert(
                "result_sha256".to_owned(),
                serde_json::Value::String("0".repeat(64)),
            );
            receipt = serde_json::to_vec(&wire).map_err(|_| {
                types::InvocationError::HostFailure(
                    "receipt negative-control encoding failed".to_owned(),
                )
            })?;
        }
        let length = u32::try_from(receipt.len()).map_err(|_| {
            types::InvocationError::HostFailure("receipt length overflow".to_owned())
        })?;
        let mut set = b"zed.sim.provider-result-receipt-set\0".to_vec();
        set.extend_from_slice(&1_u16.to_le_bytes());
        set.extend_from_slice(&1_u32.to_le_bytes());
        set.extend_from_slice(&length.to_le_bytes());
        set.extend_from_slice(&receipt);
        Ok(types::ProviderInvocationResponse {
            outputs: Vec::new(),
            receipt: set,
        })
    }
}
export!(BflLegacyPngProcessProbe);
