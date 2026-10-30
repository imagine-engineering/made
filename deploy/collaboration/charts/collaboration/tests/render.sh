#!/usr/bin/env bash
set -euo pipefail

chart=deploy/collaboration/charts/collaboration
default_output=$(mktemp)
active_output=$(mktemp)
production_output=$(mktemp)
rollback_output=$(mktemp)
ingress_output=$(mktemp)
error_output=$(mktemp)
nostr_output=$(mktemp)
nostr_ingress_output=$(mktemp)
nostr_zero_output=$(mktemp)
nostr_closed_false_output=$(mktemp)
nostr_closed_true_output=$(mktemp)
nostr_closed_disabled_false_output=$(mktemp)
trap 'rm -f "$default_output" "$active_output" "$production_output" "$rollback_output" "$ingress_output" "$error_output" "$nostr_output" "$nostr_ingress_output" "$nostr_zero_output" "$nostr_closed_false_output" "$nostr_closed_true_output" "$nostr_closed_disabled_false_output"' EXIT

active_args=(
  --set deployment.enabled=true
  --set image.tag=test
  --set runtimeSecret.name=collaboration-runtime
  --set migration.image.tag=test
  --set migration.secretName=collaboration-migration
  --set publicUrl=https://collaboration.example.invalid
  --set objectStore.endpoint=https://objects.example.invalid
  --set 'networkPolicy.postgresEgressCidrs[0]=10.10.0.0/16'
  --set 'networkPolicy.redisEgressCidrs[0]=10.20.0.0/16'
  --set 'networkPolicy.objectStoreEgressCidrs[0]=10.30.0.0/16'
)

production_args=(
  -f "$chart/values-production.yaml"
  --set image.digest=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --set runtimeSecret.name=collaboration-runtime
  --set migration.image.digest=sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  --set migration.secretName=collaboration-migration
  --set publicUrl=https://collaboration.example.invalid
  --set objectStore.endpoint=https://objects.example.invalid
  --set autoscaling.enabled=true
  --set autoscaling.minReplicas=3
  --set autoscaling.maxReplicas=10
  --set autoscaling.websocketMetric.enabled=true
  --set podMonitor.enabled=true
  --set networkPolicy.monitoring.enabled=true
  --set 'networkPolicy.monitoring.namespaceSelector.kubernetes\.io/metadata\.name=monitoring'
  --set 'networkPolicy.monitoring.podSelector.app\.kubernetes\.io/name=prometheus'
  --set 'httpRoute.parentRefs[0].name=production-gateway'
  --set 'httpRoute.parentRefs[0].namespace=gateway-system'
  --set 'httpRoute.hostnames[0]=collaboration.example.invalid'
  --set 'networkPolicy.postgresEgressCidrs[0]=10.10.0.0/16'
  --set 'networkPolicy.redisEgressCidrs[0]=10.20.0.0/16'
  --set 'networkPolicy.objectStoreEgressCidrs[0]=10.30.0.0/16'
)

helm lint "$chart" >/dev/null
helm template collaboration "$chart" >"$default_output"
if grep -q '^kind:' "$default_output"; then
  echo "expected the default-disabled chart to render no resources" >&2
  exit 1
fi

helm lint "$chart" "${active_args[@]}" >/dev/null
helm template collaboration "$chart" "${active_args[@]}" >"$active_output"

helm lint "$chart" "${production_args[@]}" >/dev/null
helm template collaboration "$chart" "${production_args[@]}" >"$production_output"

helm template collaboration "$chart" "${active_args[@]}" \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set ingress.hostname=collaboration.example.invalid \
  --set ingress.tlsSecretName=collaboration-tls \
  >"$ingress_output"

helm template collaboration "$chart" "${active_args[@]}" \
  --set push.enabled=true \
  --set push.url=https://push.example.invalid \
  --set pairing.enabled=true \
  --set pairing.url=wss://pair.example.invalid \
  --set relayMesh.enabled=true \
  --set 'relayMesh.peers[0]=wss://mesh-a.example.invalid' \
  --set 'relayMesh.peers[1]=wss://mesh-b.example.invalid' \
  --set 'networkPolicy.optionalHttpsEgressCidrs[0]=10.40.0.0/16' \
  >/dev/null

rollback_args=(
  "${production_args[@]}"
  -f "$chart/values-rollback.yaml"
  --set rollback.targetImageDigest=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  --set-string rollback.maximumSchemaVersion=20261026000100
)
helm lint "$chart" "${rollback_args[@]}" >/dev/null
helm template collaboration "$chart" "${rollback_args[@]}" >"$rollback_output"

ruby - "$active_output" "$production_output" "$rollback_output" "$ingress_output" <<'RUBY'
require "yaml"

active = YAML.load_stream(File.read(ARGV[0])).compact
production = YAML.load_stream(File.read(ARGV[1])).compact
rollback = YAML.load_stream(File.read(ARGV[2])).compact
ingress = YAML.load_stream(File.read(ARGV[3])).compact

deployment = active.find { |resource| resource["kind"] == "Deployment" }
service = active.find { |resource| resource["kind"] == "Service" }
job = active.find { |resource| resource["kind"] == "Job" }
pvc = active.find { |resource| resource["kind"] == "PersistentVolumeClaim" }
policies = active.select { |resource| resource["kind"] == "NetworkPolicy" }
raise "missing active resources" unless deployment && service && job && pvc && policies.length == 2
raise "non-public port exposed" unless service.dig("spec", "ports").map { |port| port["targetPort"] } == ["public"]
raise "Git storage is not ReadWriteMany" unless pvc.dig("spec", "accessModes") == ["ReadWriteMany"]

container = deployment.dig("spec", "template", "spec", "containers", 0)
raise "wrong liveness contract" unless container.dig("livenessProbe", "periodSeconds") == 10 && container.dig("livenessProbe", "timeoutSeconds") == 3
raise "wrong readiness contract" unless container.dig("readinessProbe", "periodSeconds") == 5 && container.dig("readinessProbe", "failureThreshold") == 3
raise "wrong startup budget" unless container.dig("startupProbe", "periodSeconds") * container.dig("startupProbe", "failureThreshold") == 120
raise "runtime can escalate" unless container.dig("securityContext", "allowPrivilegeEscalation") == false && container.dig("securityContext", "capabilities", "drop") == ["ALL"]
raise "unbounded runtime resources" unless container.dig("resources", "limits", "memory") == "2Gi" && container.dig("resources", "limits", "ephemeral-storage") == "1Gi"

environment = container["env"].to_h { |entry| [entry["name"], entry] }
required_environment = %w[
  COLLABORATION_PUBLIC_URL COLLABORATION_DATABASE_URL COLLABORATION_REDIS_URL
  COLLABORATION_OBJECT_ENDPOINT COLLABORATION_OBJECT_REGION COLLABORATION_OBJECT_BUCKET
  COLLABORATION_OBJECT_ACCESS_KEY COLLABORATION_OBJECT_SECRET_KEY
  COLLABORATION_OBJECT_ADDRESSING_STYLE COLLABORATION_GIT_REPOSITORY_PATH
  COLLABORATION_GIT_HOOK_SECRET COLLABORATION_REPLICA_COUNT COLLABORATION_PUSH_ENABLED
  COLLABORATION_PAIRING_ENABLED COLLABORATION_RELAY_MESH_ENABLED
]
raise "canonical configuration is incomplete" unless (required_environment - environment.keys).empty?
raise "database alias drift" unless environment.dig("DATABASE_URL", "valueFrom", "secretKeyRef") == environment.dig("COLLABORATION_DATABASE_URL", "valueFrom", "secretKeyRef")
raise "object key alias drift" unless environment.dig("BLOB_STORE_ACCESS_KEY", "valueFrom", "secretKeyRef") == environment.dig("COLLABORATION_OBJECT_ACCESS_KEY", "valueFrom", "secretKeyRef")
raise "object endpoint alias drift" unless environment.dig("BLOB_STORE_URL", "value") == environment.dig("COLLABORATION_OBJECT_ENDPOINT", "value")
raise "replica admission drift" unless environment.dig("COLLABORATION_REPLICA_COUNT", "value") == "2"

raise "migration hook missing" unless job.dig("metadata", "annotations", "helm.sh/hook") == "pre-install,pre-upgrade"
raise "migration is not bounded" unless job.dig("spec", "activeDeadlineSeconds") == 300 && job.dig("spec", "backoffLimit") == 2
migration_container = job.dig("spec", "template", "spec", "containers", 0)
raise "migration command drift" unless migration_container["args"] == ["up"]
raise "migration image owner drift" unless migration_container["image"] == "ghcr.io/zed-industries/collaboration-migrations:test"
migration_secret = migration_container.dig("env", 1, "valueFrom", "secretKeyRef", "name")
raise "migration aliases runtime credentials" unless migration_secret == "collaboration-migration"

production_deployment = production.find { |resource| resource["kind"] == "Deployment" }
production_container = production_deployment.dig("spec", "template", "spec", "containers", 0)
raise "production image is mutable" unless production_container["image"].end_with?("@sha256:" + "a" * 64)
production_job = production.find { |resource| resource["kind"] == "Job" }
raise "production migration image is mutable" unless production_job.dig("spec", "template", "spec", "containers", 0, "image").end_with?("@sha256:" + "c" * 64)
raise "autoscaling replica contract drift" unless production_container["env"].find { |entry| entry["name"] == "COLLABORATION_REPLICA_COUNT" }["value"] == "10"
route = production.find { |resource| resource["kind"] == "HTTPRoute" }
raise "production route is unattached" unless route && route.dig("spec", "parentRefs", 0, "name") == "production-gateway"
route_paths = route.dig("spec", "rules", 0, "matches").map { |match| match.dig("path", "value") }
raise "private metrics path exposed" if route_paths.include?("/metrics") || route_paths.include?("/") && route.dig("spec", "rules", 0, "matches", 0, "path", "type") == "PathPrefix"
hpa = production.find { |resource| resource["kind"] == "HorizontalPodAutoscaler" }
raise "autoscaling bounds drift" unless hpa && hpa.dig("spec", "minReplicas") == 3 && hpa.dig("spec", "maxReplicas") == 10 && hpa.dig("spec", "metrics").length == 2
raise "disruption budget missing" unless production.any? { |resource| resource["kind"] == "PodDisruptionBudget" }
raise "monitoring contract missing" unless production.any? { |resource| resource["kind"] == "PodMonitor" }

ingress_resource = ingress.find { |resource| resource["kind"] == "Ingress" }
raise "TLS ingress missing" unless ingress_resource && ingress_resource.dig("spec", "tls", 0, "secretName") == "collaboration-tls"
ingress_paths = ingress_resource.dig("spec", "rules", 0, "http", "paths")
raise "Ingress exposed private metrics" if ingress_paths.any? { |path| path["path"] == "/metrics" || path["path"] == "/" && path["pathType"] == "Prefix" }

rollback_deployment = rollback.find { |resource| resource["kind"] == "Deployment" }
raise "rollback rendered a migration" if rollback.any? { |resource| resource["kind"] == "Job" || resource.dig("metadata", "name")&.end_with?("-migration") }
raise "rollback target not selected" unless rollback_deployment.dig("spec", "template", "spec", "containers", 0, "image").end_with?("@sha256:" + "b" * 64)
raise "rollback annotation missing" unless rollback_deployment.dig("spec", "template", "metadata", "annotations", "collaboration.zed.dev/rollback") == "true"
rollback_pvc = rollback.find { |resource| resource["kind"] == "PersistentVolumeClaim" }
raise "rollback changed Git storage" unless rollback_pvc.dig("spec") == production.find { |resource| resource["kind"] == "PersistentVolumeClaim" }.dig("spec")
RUBY

if helm template collaboration "$chart" "${active_args[@]}" --set runtimeSecret.name= >"$error_output" 2>&1; then
  echo "expected an empty runtime secret name to fail" >&2
  exit 1
fi
grep -q 'runtimeSecret.name is required' "$error_output"

if helm template collaboration "$chart" "${active_args[@]}" --set migration.secretName= >"$error_output" 2>&1; then
  echo "expected an empty migration secret name to fail" >&2
  exit 1
fi
grep -q 'migration.secretName is required' "$error_output"

if helm template collaboration "$chart" "${active_args[@]}" --set migration.secretName=collaboration-runtime >"$error_output" 2>&1; then
  echo "expected runtime and migration credentials to remain separate" >&2
  exit 1
fi
grep -q 'runtime and DDL-capable migration credentials must use distinct secrets' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" --set image.digest= >"$error_output" 2>&1; then
  echo "expected a production tag without a digest to fail" >&2
  exit 1
fi
grep -q 'production deployment requires image.digest' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" --set migration.image.digest= >"$error_output" 2>&1; then
  echo "expected a mutable production migration image to fail" >&2
  exit 1
fi
grep -q 'production migration requires migration.image.digest' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" --set httpRoute.enabled=false >"$error_output" 2>&1; then
  echo "expected a production release without ingress to fail" >&2
  exit 1
fi
grep -q 'production deployment requires ingress.enabled or httpRoute.enabled' "$error_output"

if helm template collaboration "$chart" "${active_args[@]}" --set redis.enabled=false >"$error_output" 2>&1; then
  echo "expected a multi-replica release without Redis to fail" >&2
  exit 1
fi
grep -q 'multiple replicas require redis.enabled=true' "$error_output"

if helm template collaboration "$chart" "${active_args[@]}" --set persistence.git.accessMode=ReadWriteOnce >"$error_output" 2>&1; then
  echo "expected multi-replica ReadWriteOnce Git storage to fail" >&2
  exit 1
fi
grep -q 'multiple replicas with persistent Git storage require ReadWriteMany' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" -f "$chart/values-rollback.yaml" --set-string rollback.maximumSchemaVersion=20261026000100 >"$error_output" 2>&1; then
  echo "expected a rollback without a target digest to fail" >&2
  exit 1
fi
grep -q 'rollback.targetImageDigest is required' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" -f "$chart/values-rollback.yaml" --set rollback.targetImageDigest=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb --set-string rollback.maximumSchemaVersion=20260824000499 >"$error_output" 2>&1; then
  echo "expected an incompatible rollback schema ceiling to fail" >&2
  exit 1
fi
grep -q 'rollback target does not support the deployed schema version' "$error_output"

if helm template collaboration "$chart" "${active_args[@]}" --set unreviewedService.enabled=true >"$error_output" 2>&1; then
  echo "expected an unknown configuration key to fail schema validation" >&2
  exit 1
fi
grep -q "additional properties 'unreviewedService' not allowed" "$error_output"

nostr_args=(
  --set nostrHttp.enabled=true
  --set 'nostrHttp.routes[0].communityId=11111111-1111-4111-8111-111111111111'
  --set 'nostrHttp.routes[0].publicUrl=https://collaboration.example.invalid'
  --set nostrHttp.callsPerMinute=60
)
helm lint "$chart" "${production_args[@]}" "${nostr_args[@]}" >/dev/null
helm template collaboration "$chart" "${production_args[@]}" "${nostr_args[@]}" >"$nostr_output"
helm template collaboration "$chart" "${active_args[@]}" "${nostr_args[@]}"   --set ingress.enabled=true --set ingress.hostname=collaboration.example.invalid   --set ingress.tlsSecretName=collaboration-tls >"$nostr_ingress_output"
helm template collaboration "$chart" "${production_args[@]}" "${nostr_args[@]}"   --set nostrHttp.callsPerMinute=0 >"$nostr_zero_output"
helm template collaboration "$chart" "${production_args[@]}" "${nostr_args[@]}" --set nostrHttp.closedWebsocket=false >"$nostr_closed_false_output"
helm template collaboration "$chart" "${production_args[@]}" "${nostr_args[@]}" --set nostrHttp.closedWebsocket=true >"$nostr_closed_true_output"
helm template collaboration "$chart" "${production_args[@]}" --set nostrHttp.closedWebsocket=false >"$nostr_closed_disabled_false_output"
ruby - "$production_output" "$nostr_output" "$nostr_ingress_output" "$nostr_zero_output" "$nostr_closed_false_output" "$nostr_closed_true_output" "$nostr_closed_disabled_false_output" <<'RUBY'
require "yaml"
require "json"

disabled, enabled, ingress, zero, closed_false, closed_true, disabled_false = ARGV.map { |path| YAML.load_stream(File.read(path)).compact }
def deployment(resources)
  resources.find { |resource| resource["kind"] == "Deployment" }
end
def environment(resources)
  deployment(resources).dig("spec", "template", "spec", "containers", 0, "env").map { |entry| [entry["name"], entry] }.to_h
end
disabled_environment = environment(disabled)
raise "disabled Nostr has explicit environment" if disabled_environment.key?("COLLABORATION_NOSTR_ROUTES") || disabled_environment.key?("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE")
raise "unset closed WebSocket default became explicit" if disabled_environment.key?("COLLABORATION_NOSTR_CLOSED_WEBSOCKET") || environment(enabled).key?("COLLABORATION_NOSTR_CLOSED_WEBSOCKET")
raise "explicit false closed WebSocket binding lost" unless environment(closed_false).dig("COLLABORATION_NOSTR_CLOSED_WEBSOCKET", "value") == "false"
raise "explicit true closed WebSocket binding lost" unless environment(closed_true).dig("COLLABORATION_NOSTR_CLOSED_WEBSOCKET", "value") == "true"
raise "disabled explicit false lost" unless environment(disabled_false).dig("COLLABORATION_NOSTR_CLOSED_WEBSOCKET", "value") == "false"
configured = environment(enabled)
raise "configured canonical Nostr routes missing" unless configured.key?("COLLABORATION_NOSTR_ROUTES") && configured.key?("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE")
expected = [{"community_id" => "11111111-1111-4111-8111-111111111111", "public_url" => "https://collaboration.example.invalid"}]
raise "canonical route JSON mismatch" unless JSON.parse(configured.dig("COLLABORATION_NOSTR_ROUTES", "value")) == expected
raise "explicit quota mismatch" unless configured.dig("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE", "value") == "60"
raise "zero quota silently defaults" unless environment(zero).dig("COLLABORATION_NOSTR_API_CALLS_PER_MINUTE", "value") == "0"
raise "Redis authority changed" unless configured["COLLABORATION_REDIS_URL"] == disabled_environment["COLLABORATION_REDIS_URL"] && configured["REDIS_URL"] == disabled_environment["REDIS_URL"]
checksum = "collaboration.zed.dev/config-checksum"
raise "route config missing from rollout checksum" if deployment(disabled).dig("spec", "template", "metadata", "annotations", checksum) == deployment(enabled).dig("spec", "template", "metadata", "annotations", checksum)
raise "quota config missing from rollout checksum" if deployment(enabled).dig("spec", "template", "metadata", "annotations", checksum) == deployment(zero).dig("spec", "template", "metadata", "annotations", checksum)
raise "closed WebSocket missing from rollout checksum" if deployment(closed_false).dig("spec", "template", "metadata", "annotations", checksum) == deployment(closed_true).dig("spec", "template", "metadata", "annotations", checksum)
[closed_false, closed_true].each do |resources|
  raise "closed WebSocket changed public route exposure" unless resources.select { |entry| ["HTTPRoute", "Ingress", "Service"].include?(entry["kind"]) } == enabled.select { |entry| ["HTTPRoute", "Ingress", "Service"].include?(entry["kind"]) }
  raise "closed WebSocket changed existing admission bindings" unless environment(resources).reject { |name, _entry| name == "COLLABORATION_NOSTR_CLOSED_WEBSOCKET" } == configured
end
[disabled, enabled, zero].each do |resources|
  route = resources.find { |resource| resource["kind"] == "HTTPRoute" }
  paths = route.dig("spec", "rules", 0, "matches").map { |entry| entry["path"] }
  ["/events", "/query"].each do |path|
    matches = paths.select { |entry| entry["value"] == path }
    if resources.equal?(disabled)
      raise "disabled profile path exposed" unless matches.empty?
    else
      raise "profile route is absent or overly broad" unless matches == [{"type" => "Exact", "value" => path}]
    end
  end
  raise "private metrics exposed" if paths.any? { |entry| entry["value"] == "/metrics" || entry["value"] == "/" && entry["type"] == "PathPrefix" }
  raise "existing root/rpc routes lost" unless paths.any? { |entry| entry == {"type" => "Exact", "value" => "/"} } && paths.any? { |entry| entry["value"] == "/rpc" }
end
ingress_paths = ingress.find { |resource| resource["kind"] == "Ingress" }.dig("spec", "rules", 0, "http", "paths")
["/events", "/query"].each do |path|
  matches = ingress_paths.select { |entry| entry["path"] == path }
  raise "profile ingress is absent or overly broad" unless matches.length == 1 && matches[0]["pathType"] == "Exact"
end
raise "profile ingress exposes metrics" if ingress_paths.any? { |entry| entry["path"] == "/metrics" }
RUBY

if helm template collaboration "$chart" "${production_args[@]}" --set nostrHttp.enabled=true   --set 'nostrHttp.routes[0].communityId=11111111-1111-4111-8111-111111111111'   --set 'nostrHttp.routes[0].publicUrl=https://collaboration.example.invalid' >"$error_output" 2>&1; then
  echo "expected missing explicit Nostr quota to fail" >&2; exit 1
fi
grep -q 'callsPerMinute' "$error_output"
if helm template collaboration "$chart" "${active_args[@]}" "${nostr_args[@]}"   --set redis.enabled=false --set deployment.replicaCount=1 >"$error_output" 2>&1; then
  echo "expected Nostr without shared Redis to fail even with one replica" >&2; exit 1
fi
grep -q 'nostrHttp.enabled requires redis.enabled=true' "$error_output"
for override in   'nostrHttp.routes[0].publicUrl=http://untrusted.example.invalid'   'nostrHttp.routes[0].publicUrl=https://user@collaboration.example.invalid'   'nostrHttp.routes[0].communityId=00000000-0000-0000-0000-000000000000'   'nostrHttp.routes[0].communityId=not-a-community'   'nostrHttp.callsPerMinute=-1'   'nostrHttp.unreviewed=true'   'nostrHttp.closedWebsocket=invalid'; do
  if helm template collaboration "$chart" "${production_args[@]}" "${nostr_args[@]}" --set "$override" >"$error_output" 2>&1; then
    echo "expected unsafe Nostr configuration to fail: $override" >&2; exit 1
  fi
  grep -q 'nostrHttp' "$error_output"
done
if helm template collaboration "$chart" "${production_args[@]}" --set nostrHttp.callsPerMinute=60 >"$error_output" 2>&1; then
  echo "expected quota without enabled Nostr routes to fail" >&2; exit 1
fi
grep -q 'callsPerMinute' "$error_output"

if helm template collaboration "$chart" "${production_args[@]}" --set nostrHttp.closedWebsocket=true >"$error_output" 2>&1; then
  echo "expected closed WebSocket without enabled profile admission to fail" >&2; exit 1
fi
grep -q 'enabled' "$error_output"

echo "collaboration Helm contract checks passed"
