wit_bindgen::generate!({
    path: "../../../../sim_plugin_sdk/wit",
    world: "sim-plugin",
});

use exports::zed::sim_plugin::plugin::Guest;
use zed::sim_plugin::{host, types};

struct EchoComponent;

impl Guest for EchoComponent {
    fn manifest() -> types::ManifestProjection {
        manifest_projection()
    }

    fn create_node(node_id: String) -> Result<u64, types::InvocationError> {
        if node_id == "echo" {
            Ok(1)
        } else {
            Err(types::InvocationError::PluginFailure(format!(
                "unknown fixture node `{node_id}`"
            )))
        }
    }

    fn invoke(instance: u64) -> Result<(), types::InvocationError> {
        require(instance == 1, "unknown fixture node instance")?;
        host::check_cancelled()?;

        transfer_scalar("scalar-single-in", "scalar-single-out")?;
        transfer_scalar("scalar-list-in", "scalar-list-out")?;
        transfer_handle("tensor-single-in", "tensor-single-out")?;
        transfer_handle("tensor-list-in", "tensor-list-out")?;
        transfer_handle("artifact-single-in", "artifact-single-out")?;
        transfer_handle("artifact-list-in", "artifact-list-out")?;
        transfer_handle("model-single-in", "model-single-out")?;
        transfer_handle("model-list-in", "model-list-out")?;

        require(
            host::filesystem_read("input-root", "nested/file.bin")? == b"file",
            "filesystem fixture response changed",
        )?;
        require(
            host::provider_request(
                "demo",
                "https://demo.invalid/v1/generate",
                b"request",
                Some("secret.demo"),
            )? == b"provider",
            "provider fixture response changed",
        )?;
        require(
            host::secret_exists("secret.demo")?,
            "fixture secret is unavailable",
        )?;
        require(
            host::clock_now("workflow")? == 1_234,
            "fixture clock response changed",
        )?;
        require(
            host::random_bytes("sampler", 8)?.len() == 8,
            "fixture random response changed",
        )?;
        require(
            host::model_open("zed-asset://model/fixture.json")? != 0,
            "fixture model handle is invalid",
        )?;

        let transaction = host::output_begin("outputs", "guest.bin")?;
        host::output_write(transaction, b"guest-output")?;
        require(
            !host::output_commit(transaction)?.is_empty(),
            "fixture output identifier is empty",
        )?;
        host::log("info", "no-WASI echo fixture invoked")?;
        host::ui_set("panel.demo", br#"{"invoked":true}"#)?;
        host::route_respond("route.demo", 200, b"guest-route")?;
        host::check_cancelled()?;
        Ok(())
    }

    fn cancel(instance: u64, _reason: types::CancelReason) -> Result<(), types::InvocationError> {
        require(instance == 1, "unknown fixture node instance")
    }

    fn drop_node(_instance: u64) {}
}

fn transfer_scalar(input: &str, output: &str) -> Result<(), types::InvocationError> {
    let state = host::get_input_state(input)?;
    require(
        state.family == types::ValueFamily::Scalar,
        "scalar input family changed",
    )?;
    for index in 0..state.length {
        let value = host::read_scalar_input(input, index)?;
        let handle = host::create_output_value(&value)?;
        require(
            !host::read_handle(handle.clone())?.abi_bytes.is_empty(),
            "created scalar value is empty",
        )?;
        host::push_output(output, handle)?;
    }
    host::finish_output(output, state.present)
}

fn transfer_handle(input: &str, output: &str) -> Result<(), types::InvocationError> {
    let state = host::get_input_state(input)?;
    require(
        state.family != types::ValueFamily::Scalar,
        "handle input family changed",
    )?;
    for index in 0..state.length {
        let handle = host::take_input(input, index)?;
        require(
            !host::read_handle(handle.clone())?.abi_bytes.is_empty(),
            "transferred handle value is empty",
        )?;
        host::push_output(output, handle)?;
    }
    host::finish_output(output, state.present)
}

fn require(condition: bool, message: &str) -> Result<(), types::InvocationError> {
    if condition {
        Ok(())
    } else {
        Err(types::InvocationError::PluginFailure(message.to_owned()))
    }
}

fn manifest_projection() -> types::ManifestProjection {
    types::ManifestProjection {
        component_world: "zed:sim-plugin@1.0.0".to_owned(),
        schema_version: 1,
        identifier: "test.echo-plugin".to_owned(),
        plugin_version: version(1, 2, 3),
        api: types::ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                "capabilities.transactional".to_owned(),
                "handles.revocation".to_owned(),
                "legacy.non-destructive".to_owned(),
                "ports.list".to_owned(),
            ],
        },
        nodes: vec![types::Node {
            id: "echo".to_owned(),
            version: version(1, 0, 0),
            display_name: "Echo".to_owned(),
            category: "test".to_owned(),
            ports: manifest_ports(),
            determinism: types::DeterminismPolicy::Deterministic,
            cache: types::CachePolicy::InputIdentity,
            effects: types::EffectPolicy::Transactional,
        }],
        capabilities: capability_requests(),
        ui: vec![types::UiContribution {
            id: "panel.demo".to_owned(),
            surface: "node-panel".to_owned(),
            state_schema: "{\"type\":\"object\"}".to_owned(),
        }],
        routes: vec![types::RouteDeclaration {
            id: "route.demo".to_owned(),
            method: "POST".to_owned(),
            path: "/plugins/test/echo".to_owned(),
            maximum_request_bytes: 4_096,
            maximum_response_bytes: 4_096,
        }],
        legacy_mappings: vec![types::LegacyMapping {
            legacy_identifier: "LegacyEcho".to_owned(),
            node_id: "echo".to_owned(),
            node_version: version(1, 0, 0),
        }],
    }
}

fn manifest_ports() -> Vec<types::Port> {
    let mut ports = Vec::new();
    for (family, type_id, presence, serialization) in [
        (
            types::ValueFamily::Scalar,
            "sim:string@1",
            types::PortPresence::Required,
            types::PortSerialization::Inline,
        ),
        (
            types::ValueFamily::Tensor,
            "sim:image@1",
            types::PortPresence::Required,
            types::PortSerialization::Handle,
        ),
        (
            types::ValueFamily::Artifact,
            "sim:svg@1",
            types::PortPresence::Optional,
            types::PortSerialization::ArtifactReference,
        ),
        (
            types::ValueFamily::Model,
            "sim:model@1",
            types::PortPresence::Hidden,
            types::PortSerialization::Handle,
        ),
    ] {
        let (name, output_presence) = match family {
            types::ValueFamily::Scalar => ("scalar", types::PortPresence::Required),
            types::ValueFamily::Tensor => ("tensor", types::PortPresence::Required),
            types::ValueFamily::Artifact => ("artifact", types::PortPresence::Optional),
            types::ValueFamily::Model => ("model", types::PortPresence::Required),
        };
        ports.push(port(
            &format!("{name}-single-in"),
            types::PortDirection::Input,
            type_id,
            types::PortCardinality::Singular,
            presence,
            false,
            serialization,
        ));
        ports.push(port(
            &format!("{name}-single-out"),
            types::PortDirection::Output,
            type_id,
            types::PortCardinality::Singular,
            output_presence,
            false,
            serialization,
        ));
        ports.push(port(
            &format!("{name}-list-in"),
            types::PortDirection::Input,
            type_id,
            types::PortCardinality::List,
            types::PortPresence::Optional,
            true,
            serialization,
        ));
        ports.push(port(
            &format!("{name}-list-out"),
            types::PortDirection::Output,
            type_id,
            types::PortCardinality::List,
            types::PortPresence::Optional,
            false,
            serialization,
        ));
    }
    ports
}

fn port(
    id: &str,
    direction: types::PortDirection,
    type_id: &str,
    cardinality: types::PortCardinality,
    presence: types::PortPresence,
    lazy: bool,
    serialization: types::PortSerialization,
) -> types::Port {
    types::Port {
        id: id.to_owned(),
        name: id.to_owned(),
        direction,
        type_id: type_id.to_owned(),
        cardinality,
        presence,
        hidden: presence == types::PortPresence::Hidden,
        lazy,
        default: None,
        serialization,
        accepted_legacy_names: if id == "scalar-single-in" {
            vec!["legacy_scalar".to_owned()]
        } else {
            Vec::new()
        },
    }
}

fn capability_requests() -> Vec<types::CapabilityRequest> {
    [
        (types::CapabilityKind::Filesystem, "input-root"),
        (types::CapabilityKind::Filesystem, "model-root"),
        (
            types::CapabilityKind::NetworkProvider,
            "demo|https://demo.invalid/v1/generate",
        ),
        (types::CapabilityKind::Secret, "secret.demo"),
        (types::CapabilityKind::Clock, "workflow"),
        (types::CapabilityKind::Randomness, "sampler"),
        (
            types::CapabilityKind::Model,
            "zed-asset://model/fixture.json",
        ),
        (types::CapabilityKind::TransactionalOutput, "outputs"),
        (
            types::CapabilityKind::TransactionalOutput,
            "output-transaction",
        ),
        (types::CapabilityKind::SanitizedLog, "info"),
        (types::CapabilityKind::DeclarativeUi, "panel.demo"),
        (types::CapabilityKind::Route, "route.demo"),
    ]
    .into_iter()
    .map(|(kind, scope)| types::CapabilityRequest {
        kind,
        scope: scope.to_owned(),
        quota: types::CapabilityQuota {
            maximum_operations: 16,
            maximum_request_bytes: 4_096,
            maximum_response_bytes: 4_096,
            maximum_total_bytes: 32_768,
            maximum_handles: 8,
            timeout_milliseconds: 5_000,
        },
    })
    .collect()
}

fn version(major: u16, minor: u16, patch: u16) -> types::ApiVersion {
    types::ApiVersion {
        major,
        minor,
        patch,
    }
}

export!(EchoComponent);
