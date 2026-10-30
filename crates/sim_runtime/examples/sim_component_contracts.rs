use serde_json::{Value, json};
use sim_plugin_sdk::{ProviderBindingClaim, ProviderBindingSet, TypeRegistry};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::{self, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let registry = sim_runtime::generated_native_node_registry_projection(None)?;
    let mut contracts = BTreeMap::new();
    let mut binding_sets = BTreeMap::new();
    for path in std::env::args_os().skip(1) {
        let fixture: Value = serde_json::from_slice(&fs::read(path)?)?;
        let namespace = fixture
            .get("namespace")
            .and_then(Value::as_str)
            .ok_or("missing namespace")?;
        let claims = fixture
            .get("bindings")
            .and_then(Value::as_array)
            .ok_or("missing binding claims")?;
        let mut bindings = Vec::new();
        for claim in claims {
            let mut claim: ProviderBindingClaim = serde_json::from_value(claim.clone())?;
            // Synthetic multi-contract fixtures deliberately use a descriptor absent from the registry.
            if let Some(digest) = registry.provider_binding_contract_sha256(
                &claim.node_id,
                &claim.transport_schema.to_string(),
                &claim.materializer_schema.to_string(),
            )? {
                contracts.insert(claim.contract_sha256.clone(), digest.clone());
                claim.contract_sha256 = digest;
            }
            bindings.push(claim);
        }
        let binding = ProviderBindingSet {
            schema_version: 1,
            implementation_namespace: namespace.to_owned(),
            bindings_sha256: String::new(),
            bindings,
        };
        let old = fixture
            .get("bindings_sha256")
            .and_then(Value::as_str)
            .ok_or("missing binding digest")?;
        binding_sets.insert(old.to_owned(), binding.canonical_bindings_sha256()?);
    }
    let output = json!({"contracts": contracts, "binding_sets": binding_sets,
        "type_registry_projection": TypeRegistry::built_in()?.canonical_projection()});
    serde_json::to_writer_pretty(io::stdout().lock(), &output)?;
    io::stdout().lock().write_all(b"\n")?;
    Ok(())
}
