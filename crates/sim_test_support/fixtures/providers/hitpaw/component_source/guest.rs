wit_bindgen::generate!({
    path: "../../../../../sim_plugin_sdk/wit/provider-v2",
    world: "sim-provider-plugin",
    with: { "zed:sim-plugin/types@1.0.0": generate },
});

use exports::zed::sim_provider_plugin::provider_node;
use zed::sim_plugin::types as plugin;
use zed::sim_provider_plugin::{invocation_input_host, provider_streaming_host, types};

const NODE: &str = "HitPawGeneralImageEnhance";
const NAMESPACE: &str = "zed.sim.provider.hitpaw";
const ENDPOINT: &str = "https://api.comfy.org/proxy/hitpaw/api/photo-enhancer";
const CONTRACT_SHA256: &str = env!("SIM_HITPAW_NATIVE_CONTRACT_SHA256");
const BINDINGS_SHA256: &str = env!("SIM_HITPAW_BINDINGS_SHA256");

struct HitPawImageInputAdmissionProbe;

impl provider_node::Guest for HitPawImageInputAdmissionProbe {
    fn manifest() -> types::ManifestV2 {
        types::ManifestV2 {
            schema_version: 2,
            component_world: "zed:sim-provider-plugin@2.0.0".into(),
            manifest: manifest(),
            provider_binding: plugin::ProviderBindingSet {
                schema_version: 1,
                implementation_namespace: NAMESPACE.into(),
                bindings_sha256: BINDINGS_SHA256.into(),
                bindings: vec![plugin::ProviderBindingClaim {
                    feature_id: "COMFY-NODE-0226".into(),
                    node_id: NODE.into(),
                    contract_sha256: CONTRACT_SHA256.into(),
                    transport_schema: "zed:sim-provider-transport@1".into(),
                    materializer_schema: "zed:sim-provider-materializer@1".into(),
                }],
            },
            streaming: types::StreamingContract {
                methods: vec![types::HttpMethod::Post],
                maximum_headers: 8,
                maximum_header_bytes: 16_384,
                maximum_request_body_bytes: 1_048_576,
                maximum_response_body_bytes: 16_777_216,
                maximum_chunk_bytes: 65_536,
                maximum_ndjson_line_bytes: 65_536,
                maximum_wait_milliseconds: 1_000,
                maximum_uploads: 0,
                maximum_upload_body_bytes: 0,
                maximum_cost_requests: 0,
                maximum_progress_total: 100,
                uploads: false,
                cost_requests: false,
            },
        }
    }

    fn invoke(
        context: types::InvocationContext,
        node_id: String,
    ) -> Result<types::InvocationResult, types::StreamError> {
        if node_id != NODE {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        invocation_input_host::check_cancelled()
            .map_err(|error| input_error(error, types::StreamError::HostFailure))?;
        let model = scalar("model")?;
        if !matches!(
            model.get("value").and_then(serde_json::Value::as_str),
            Some("generative" | "generative_portrait")
        ) {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let scale = scalar("upscale_factor")?;
        if !matches!(
            scale.get("value").and_then(serde_json::Value::as_u64),
            Some(1 | 2 | 4)
        ) {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let automatic = scalar("auto_downscale")?;
        if automatic
            .get("value")
            .and_then(serde_json::Value::as_bool)
            .is_none()
        {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let state = invocation_input_host::get_input_state("image")
            .map_err(|error| input_error(error, types::StreamError::InvalidUpload))?;
        if !state.present || state.length != 1 {
            return Err(types::StreamError::InvalidUpload);
        }
        let handle = invocation_input_host::take_input("image", 0)
            .map_err(|error| input_error(error, types::StreamError::InvalidUpload))?;
        let image = invocation_input_host::read_handle(handle)
            .map_err(|error| input_error(error, types::StreamError::InvalidUpload))?;
        if image.type_id != "sim:image@1" || image.family != plugin::ValueFamily::Tensor {
            return Err(types::StreamError::InvalidUpload);
        }
        let value: serde_json::Value = serde_json::from_slice(&image.abi_bytes)
            .map_err(|_| types::StreamError::InvalidUpload)?;
        let shape = value
            .pointer("/representation/value/descriptor/shape")
            .and_then(serde_json::Value::as_array)
            .ok_or(types::StreamError::InvalidUpload)?;
        if shape.len() != 4
            || shape
                .iter()
                .any(|dimension| dimension.as_u64().is_none_or(|dimension| dimension == 0))
        {
            return Err(types::StreamError::InvalidUpload);
        }
        if invocation_input_host::media_input_count(handle)
            .map_err(|error| input_error(error, types::StreamError::InvalidUpload))?
            != 1
        {
            return Err(types::StreamError::InvalidUpload);
        }
        let media = invocation_input_host::read_media_input_at(handle, 0)
            .map_err(|error| input_error(error, types::StreamError::InvalidUpload))?;
        // Digest verification and PNG preparation belong to the checked input host.
        if media.media_type != "image/png"
            || !media.bytes.starts_with(b"\x89PNG\r\n\x1a\n")
            || !digest_shape(&media.source_value_sha256)
            || !digest_shape(&media.content_sha256)
        {
            return Err(types::StreamError::InvalidUpload);
        }
        let authorization = if invocation_input_host::secret_exists("sim_org.auth_token")
            .map_err(|error| input_error(error, types::StreamError::InvalidRequestAuthority))?
        {
            (
                "sim_org.auth_token",
                header("Authorization", "host-capability:auth_token_sim_org"),
            )
        } else if invocation_input_host::secret_exists("sim_org.api_key")
            .map_err(|error| input_error(error, types::StreamError::InvalidRequestAuthority))?
        {
            (
                "sim_org.api_key",
                header("X-API-KEY", "host-capability:api_key_sim_org"),
            )
        } else {
            return Err(types::StreamError::InvalidRequestAuthority);
        };
        let handle = provider_streaming_host::start_request(
            context,
            &types::RequestHead {
                endpoint: ENDPOINT.into(),
                secret_id: Some(authorization.0.into()),
                method: types::HttpMethod::Post,
                headers: vec![
                    header("Accept", "application/json"),
                    authorization.1,
                    header("Comfy-Env", "host"),
                    header("Comfy-Usage-Source", "comfyui-api"),
                    header("Content-Type", "application/json"),
                ],
                declared_body_bytes: None,
            },
        )?;
        provider_streaming_host::check_cancelled(handle)?;
        // A non-cancelled probe cannot imply upload, caller cost authority, or IMAGE output support.
        Err(types::StreamError::InvalidContract)
    }
}

fn input_error(error: plugin::InvocationError, fallback: types::StreamError) -> types::StreamError {
    match error {
        plugin::InvocationError::Cancelled => types::StreamError::Cancelled,
        plugin::InvocationError::TimedOut => types::StreamError::TimedOut,
        _ => fallback,
    }
}

fn scalar(port: &str) -> Result<serde_json::Value, types::StreamError> {
    let input = invocation_input_host::read_scalar_input(port, 0)
        .map_err(|error| input_error(error, types::StreamError::InvalidRequestAuthority))?;
    let value: serde_json::Value = serde_json::from_slice(&input.abi_bytes)
        .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
    value
        .pointer("/representation/value")
        .cloned()
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn digest_shape(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn header(name: &str, value: &str) -> types::Header {
    types::Header {
        name: name.into(),
        value: value.into(),
    }
}

fn version() -> plugin::ApiVersion {
    plugin::ApiVersion {
        major: 1,
        minor: 0,
        patch: 0,
    }
}

fn port(id: &str, type_id: &str, output: bool) -> plugin::Port {
    plugin::Port {
        id: id.into(),
        name: id.into(),
        direction: if output {
            plugin::PortDirection::Output
        } else {
            plugin::PortDirection::Input
        },
        type_id: type_id.into(),
        cardinality: plugin::PortCardinality::Singular,
        presence: plugin::PortPresence::Required,
        hidden: false,
        lazy: false,
        default: None,
        serialization: if type_id == "sim:image@1" {
            plugin::PortSerialization::Handle
        } else {
            plugin::PortSerialization::Inline
        },
        accepted_legacy_names: Vec::new(),
    }
}

fn manifest() -> plugin::ManifestProjection {
    plugin::ManifestProjection {
        component_world: "zed:sim-provider-plugin@1.0.0".into(),
        schema_version: 1,
        identifier: NAMESPACE.into(),
        plugin_version: version(),
        api: plugin::ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                "provider.bindings.v1".into(),
                "provider.streaming.v2".into(),
            ],
        },
        nodes: vec![plugin::Node {
            id: NODE.into(),
            version: version(),
            display_name: "HitPaw General Image Enhance".into(),
            category: "partner/image/HitPaw".into(),
            ports: vec![
                port("model", "sim:combo@1", false),
                port("image", "sim:image@1", false),
                port("upscale_factor", "sim:combo@1", false),
                port("auto_downscale", "sim:boolean@1", false),
                port("output_0", "sim:image@1", true),
            ],
            determinism: plugin::DeterminismPolicy::External,
            cache: plugin::CachePolicy::Never,
            effects: plugin::EffectPolicy::Provider,
        }],
        capabilities: vec![
            capability(
                plugin::CapabilityKind::NetworkProvider,
                "zed.sim.provider.hitpaw|https://api.comfy.org/proxy/hitpaw/api/photo-enhancer",
            ),
            capability(plugin::CapabilityKind::Secret, "sim_org.auth_token"),
            capability(plugin::CapabilityKind::Secret, "sim_org.api_key"),
        ],
        ui: Vec::new(),
        routes: vec![plugin::RouteDeclaration {
            id: "COMFY-API-EXT-0049".into(),
            method: "POST".into(),
            path: "/proxy/hitpaw/api/photo-enhancer".into(),
            maximum_request_bytes: 1_048_576,
            maximum_response_bytes: 16_777_216,
        }],
        legacy_mappings: Vec::new(),
    }
}

fn capability(kind: plugin::CapabilityKind, scope: &str) -> plugin::CapabilityRequest {
    plugin::CapabilityRequest {
        kind,
        scope: scope.into(),
        quota: plugin::CapabilityQuota {
            maximum_operations: 1,
            maximum_request_bytes: 1_048_576,
            maximum_response_bytes: 16_777_216,
            maximum_total_bytes: 17_825_792,
            maximum_handles: 1,
            timeout_milliseconds: 60_000,
        },
    }
}

export!(HitPawImageInputAdmissionProbe);
