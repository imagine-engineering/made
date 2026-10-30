#![cfg(unix)]

use std::{
    fs,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use remote::{
    agent_provider_discovery::{
        AgentProviderDiscoveryReport, AgentProviderSearchDirectory, AgentProviderTrust,
    },
    agent_provider_lifecycle::{
        AgentProviderDeployDisposition, AgentProviderDeployInput, AgentProviderExecutionApproval,
        AgentProviderLifecycleError, RemoteAgentLifecycleState, RemoteAgentProviderLifecycle,
    },
    agent_provider_protocol::{
        AgentProviderCancellation, AgentProviderOperation, AgentProviderProtocolError,
        AgentProviderResponse, invoke_agent_provider,
    },
};
use serde_json::{Value, json};

const AGENT_IDENTITY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const PRIVATE_KEY_NSEC: &str = "nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5";
const AUTH_TAG: &str = "fixture-authorization-secret";
const ENVIRONMENT_SECRET: &str = "fixture-environment-secret";

const KUBERNETES_INFO_RESPONSE: &str = r#"{"ok":true,"name":"kubernetes","version":"0.1.0","protocol_version":1,"description":"Runs agents as pods in a Kubernetes cluster","config_schema":{"type":"object","properties":{"context":{"type":"string"},"namespace":{"type":"string","default":"buzz-agents-fixture"},"image":{"type":"string","default":"ghcr.io/block/buzz-sprig:sha-fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"cpu_request":{"type":"string","default":"1"},"memory_request":{"type":"string","default":"2Gi"},"cpu_limit":{"type":"string","default":"2"},"memory_limit":{"type":"string","default":"4Gi"},"inactivity_seconds":{"type":"integer","default":7200},"service_account":{"type":"string"}},"required":["namespace","image"]}}"#;

struct ProviderFixture {
    _directory: tempfile::TempDir,
    path: PathBuf,
    discovery: AgentProviderDiscoveryReport,
    approval: AgentProviderExecutionApproval,
}

fn provider_fixture(script: &str) -> ProviderFixture {
    let directory = tempfile::tempdir().expect("provider fixture directory");
    let path = directory.path().join("buzz-backend-kubernetes");
    fs::write(&path, script).expect("provider fixture script");
    let mut permissions = fs::metadata(&path)
        .expect("provider fixture metadata")
        .permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&path, permissions).expect("provider fixture permissions");
    let discovery = AgentProviderDiscoveryReport::discover([AgentProviderSearchDirectory::new(
        directory.path(),
        AgentProviderTrust::Untrusted,
    )]);
    let approval = AgentProviderExecutionApproval {
        executable: discovery
            .resolve("kubernetes")
            .expect("Kubernetes provider discovery")
            .executable_reference(),
        allow_untrusted: true,
    };
    ProviderFixture {
        _directory: directory,
        path,
        discovery,
        approval,
    }
}

fn shell_literal(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn kubernetes_agent() -> Value {
    json!({
        "agent_args": [],
        "agent_command": "goose",
        "auth_tag": AUTH_TAG,
        "env_vars": {
            "USER_KEY": ENVIRONMENT_SECRET
        },
        "idle_timeout_seconds": null,
        "launch": {
            "args": ["acp"],
            "command": "goose",
            "env": {
                "GOOSE_MODEL": "gpt-5",
                "GOOSE_PROVIDER": "openai",
                "USER_KEY": ENVIRONMENT_SECRET
            },
            "owner_pubkey": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "policy_env": {
                "BUZZ_ACP_AGENTS": "10",
                "BUZZ_ACP_DISPLAY_NAME": "worker",
                "BUZZ_ACP_LAZY_POOL": "true",
                "BUZZ_ACP_MODEL": "gpt-5",
                "BUZZ_ACP_RELAY_OBSERVER": "true",
                "BUZZ_ACP_SESSION_TITLE": "worker",
                "GOOSE_MODE": "auto"
            }
        },
        "max_turn_duration_seconds": null,
        "model": "gpt-5",
        "name": "worker",
        "parallelism": 10,
        "private_key_nsec": PRIVATE_KEY_NSEC,
        "provider": "openai",
        "relay_url": "wss://relay.example",
        "respond_to": "allowlist",
        "respond_to_allowlist": [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ],
        "system_prompt": null,
        "turn_timeout_seconds": 300
    })
}

fn kubernetes_provider_config() -> Value {
    json!({
        "namespace": "buzz-agents-test",
        "image": "ghcr.io/block/buzz-sprig@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "inactivity_seconds": 3600
    })
}

fn deploy_input(operation_id: &str, work_directory: &Path) -> AgentProviderDeployInput {
    AgentProviderDeployInput {
        operation_id: operation_id.to_owned(),
        work_directory: work_directory.to_owned(),
        agent: kubernetes_agent(),
        provider_config: kubernetes_provider_config(),
    }
}

fn parse_request_log(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("provider request log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("provider request JSON"))
        .collect()
}

fn contains_secret_like_key(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.iter().any(|(key, value)| {
            let normalized = key.to_ascii_lowercase();
            normalized.contains("secret")
                || normalized.contains("password")
                || normalized.contains("token")
                || normalized.contains("credential")
                || contains_secret_like_key(value)
        }),
        Value::Array(values) => values.iter().any(contains_secret_like_key),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
    }
}

#[gpui::test]
async fn agent_provider_conformance_l1_l3_launch_and_exactly_once_deployment(
    background_executor: gpui::BackgroundExecutor,
) {
    background_executor.allow_parking();
    let work_directory = tempfile::tempdir().expect("work directory");
    let request_log = work_directory.path().join("requests.jsonl");
    let script = format!(
        "#!/bin/sh\nIFS= read -r request\nprintf '%s\\n' \"$request\" >> {}\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{}' ;;\n  *'\"op\":\"deploy\"'*) printf '%s' '{{\"ok\":true,\"agent_id\":\"buzz-agent-kubernetes-fixture\"}}' ;;\n  *) exit 9 ;;\nesac\n",
        shell_literal(&request_log),
        KUBERNETES_INFO_RESPONSE,
    );
    let fixture = provider_fixture(&script);
    let candidate = fixture
        .discovery
        .resolve("kubernetes")
        .expect("resolve discovered provider");
    let info = invoke_agent_provider(
        candidate,
        work_directory.path(),
        AgentProviderOperation::Info,
        &json!({"op": "info", "request_id": "kube-info"}),
        &background_executor,
    )
    .await
    .expect("Kubernetes info fixture");
    let AgentProviderResponse::Info(info) = info else {
        panic!("expected provider info")
    };
    assert_eq!(info.name, "kubernetes");
    assert_eq!(info.protocol_version, 1);
    assert_eq!(
        info.config_schema["required"],
        json!(["namespace", "image"])
    );
    assert_eq!(
        info.config_schema["properties"]["inactivity_seconds"]["default"],
        7200
    );
    assert!(!contains_secret_like_key(&info.config_schema));

    let mut lifecycle = RemoteAgentProviderLifecycle::new("kubernetes", AGENT_IDENTITY)
        .expect("Kubernetes lifecycle");
    let first = lifecycle
        .deploy(
            &fixture.discovery,
            &fixture.approval,
            deploy_input("kube-deploy-1", work_directory.path()),
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect("Kubernetes fixture deployment");
    let AgentProviderDeployDisposition::Deployed(deployment) = first else {
        panic!("first deployment must be new")
    };
    assert_eq!(deployment.agent_id, "buzz-agent-kubernetes-fixture");

    let requests = parse_request_log(&request_log);
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0]["op"], "info");
    assert_eq!(requests[1]["op"], "info");
    for request in &requests[..2] {
        let encoded = request.to_string();
        assert!(!encoded.contains(PRIVATE_KEY_NSEC));
        assert!(!encoded.contains(AUTH_TAG));
        assert!(!encoded.contains(ENVIRONMENT_SECRET));
    }
    let deploy = &requests[2];
    assert_eq!(deploy["op"], "deploy");
    assert_eq!(deploy["agent"], kubernetes_agent());
    assert_eq!(deploy["provider_config"], kubernetes_provider_config());
    assert_eq!(deploy["agent"]["launch"]["command"], "goose");
    assert_eq!(deploy["agent"]["launch"]["args"], json!(["acp"]));
    assert!(deploy["agent"]["launch"]["owner_pubkey"].is_string());
    assert!(
        deploy["agent"]["launch"]["env"]
            .get("BUZZ_ACP_NO_PRESENCE")
            .is_none()
    );
    assert_eq!(deploy["provider_config"]["inactivity_seconds"], 3600);
    assert!(
        deploy["provider_config"]["image"]
            .as_str()
            .is_some_and(|image| image.contains("@sha256:"))
    );

    let second = lifecycle
        .deploy(
            &fixture.discovery,
            &fixture.approval,
            deploy_input("kube-deploy-replay", work_directory.path()),
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect("idempotent Kubernetes deployment");
    assert_eq!(
        second,
        AgentProviderDeployDisposition::AlreadyDeployed(deployment)
    );
    assert_eq!(parse_request_log(&request_log).len(), 3);
    let inspection = lifecycle.inspect();
    assert!(!inspection.provider_inspection_supported);
    assert!(!inspection.provider_termination_supported);
}

#[gpui::test]
async fn agent_provider_conformance_rejects_pre_secret_and_malicious_output(
    background_executor: gpui::BackgroundExecutor,
) {
    background_executor.allow_parking();
    let work_directory = tempfile::tempdir().expect("work directory");
    let deploy_marker = work_directory.path().join("deploy-reached");
    let absent_version = format!(
        "#!/bin/sh\nIFS= read -r request\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{{\"ok\":true,\"name\":\"kubernetes\",\"version\":\"0.1.0\",\"description\":\"missing version\",\"config_schema\":{{}}}}' ;;\n  *) touch {}; printf '%s' '{{\"ok\":true,\"agent_id\":\"must-not-launch\"}}' ;;\nesac\n",
        shell_literal(&deploy_marker),
    );
    let fixture = provider_fixture(&absent_version);
    let mut lifecycle = RemoteAgentProviderLifecycle::new("kubernetes", AGENT_IDENTITY)
        .expect("Kubernetes lifecycle");
    let error = lifecycle
        .deploy(
            &fixture.discovery,
            &fixture.approval,
            deploy_input("missing-version", work_directory.path()),
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect_err("missing protocol version must fail before deploy");
    assert!(matches!(
        error,
        AgentProviderLifecycleError::ProviderOperation {
            operation: AgentProviderOperation::Info,
            source: AgentProviderProtocolError::MalformedResponse { .. }
        }
    ));
    assert!(!deploy_marker.exists());
    assert_eq!(lifecycle.inspect().state, RemoteAgentLifecycleState::Ready);

    let echoed = format!("{PRIVATE_KEY_NSEC} {AUTH_TAG} {ENVIRONMENT_SECRET}");
    let secret_echo = format!(
        "#!/bin/sh\nIFS= read -r request\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{}' ;;\n  *) printf '%s' '{{\"ok\":false,\"error\":\"{}\"}}' ;;\nesac\n",
        KUBERNETES_INFO_RESPONSE, echoed,
    );
    let fixture = provider_fixture(&secret_echo);
    let mut lifecycle = RemoteAgentProviderLifecycle::new("kubernetes", AGENT_IDENTITY)
        .expect("Kubernetes lifecycle");
    let error = lifecycle
        .deploy(
            &fixture.discovery,
            &fixture.approval,
            deploy_input("secret-echo", work_directory.path()),
            &AgentProviderCancellation::default(),
            &background_executor,
        )
        .await
        .expect_err("secret-bearing provider rejection must fail");
    let diagnostic = error.to_string();
    assert!(!diagnostic.contains(PRIVATE_KEY_NSEC));
    assert!(!diagnostic.contains(AUTH_TAG));
    assert!(!diagnostic.contains(ENVIRONMENT_SECRET));
    assert!(diagnostic.contains("[REDACTED]"));
    assert!(matches!(
        lifecycle.inspect().state,
        RemoteAgentLifecycleState::DeploymentUncertain { .. }
    ));
}

#[gpui::test]
async fn agent_provider_conformance_cancellation_cleans_process_tree_and_staging(
    background_executor: gpui::BackgroundExecutor,
) {
    background_executor.allow_parking();
    let work_directory = tempfile::tempdir().expect("work directory");
    let process_marker = work_directory.path().join("provider-processes");
    let script = format!(
        "#!/bin/sh\nIFS= read -r request\ncase \"$request\" in\n  *'\"op\":\"info\"'*) printf '%s' '{}' ;;\n  *) sleep 30 & child=$!; printf '%s %s %s\\n' \"$0\" \"$$\" \"$child\" > {}; wait \"$child\" ;;\nesac\n",
        KUBERNETES_INFO_RESPONSE,
        shell_literal(&process_marker),
    );
    let fixture = provider_fixture(&script);
    let cancellation = AgentProviderCancellation::default();
    let cancellation_thread = {
        let cancellation = cancellation.clone();
        let process_marker = process_marker.clone();
        std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(2);
            while !process_marker.exists() && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert!(process_marker.exists(), "deploy fixture did not start");
            cancellation.cancel();
        })
    };
    let mut lifecycle = RemoteAgentProviderLifecycle::new("kubernetes", AGENT_IDENTITY)
        .expect("Kubernetes lifecycle");
    let error = lifecycle
        .deploy(
            &fixture.discovery,
            &fixture.approval,
            deploy_input("cancel-cleanup", work_directory.path()),
            &cancellation,
            &background_executor,
        )
        .await
        .expect_err("cancelled deployment must fail");
    cancellation_thread.join().expect("cancellation thread");
    assert!(matches!(
        error,
        AgentProviderLifecycleError::ProviderOperation {
            operation: AgentProviderOperation::Deploy,
            source: AgentProviderProtocolError::Cancelled {
                operation: AgentProviderOperation::Deploy
            }
        }
    ));

    let marker = fs::read_to_string(&process_marker).expect("provider process marker");
    let mut fields = marker.split_whitespace();
    let staged_path = PathBuf::from(fields.next().expect("staged provider path"));
    let provider_pid = fields.next().expect("provider pid");
    let child_pid = fields.next().expect("provider child pid");
    assert!(fields.next().is_none());
    let deadline = background_executor.now() + Duration::from_secs(2);
    while (process_exists(provider_pid).await || process_exists(child_pid).await)
        && background_executor.now() < deadline
    {
        background_executor.timer(Duration::from_millis(10)).await;
    }
    assert!(!process_exists(provider_pid).await);
    assert!(!process_exists(child_pid).await);
    assert!(!staged_path.exists());
    assert!(staged_path.parent().is_some_and(|parent| !parent.exists()));
    assert!(fixture.path.exists());
    assert!(matches!(
        lifecycle.inspect().state,
        RemoteAgentLifecycleState::DeploymentUncertain { .. }
    ));
}

async fn process_exists(pid: &str) -> bool {
    smol::process::Command::new("/bin/kill")
        .args(["-0", pid])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}
