#!/usr/bin/env bash
set -euo pipefail

chart=deploy/collaboration/push-gateway
default_output=$(mktemp)
active_output=$(mktemp)
production_output=$(mktemp)
rollback_output=$(mktemp)
error_output=$(mktemp)
trap 'rm -f "$default_output" "$active_output" "$production_output" "$rollback_output" "$error_output"' EXIT

active_args=(
  --set deployment.enabled=true
  --set image.tag=test
  --set runtimeSecret.name=push-runtime
  --set migration.secretName=push-migration
  --set publicDeliveryUrl=https://push.example.invalid/v1/deliveries/apns
  --set appAttest.appIdentifier=TEAM.example.app
  --set appAttest.rootCertificateSecretName=app-attest-root
  --set profiles.production.credentialSecretName=apns-production-key
  --set profiles.production.configurationSecretName=apns-production-config
)

production_args=(
  -f "$chart/values-production.yaml"
  --set image.digest=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --set runtimeSecret.name=push-runtime
  --set migration.secretName=push-migration
  --set publicDeliveryUrl=https://push.example.invalid/v1/deliveries/apns
  --set appAttest.appIdentifier=TEAM.example.app
  --set appAttest.rootCertificateSecretName=app-attest-root
  --set profiles.production.credentialSecretName=apns-production-key
  --set profiles.production.configurationSecretName=apns-production-config
  --set 'httpRoute.parentRefs[0].name=production-gateway'
  --set 'httpRoute.parentRefs[0].namespace=gateway-system'
  --set 'httpRoute.hostnames[0]=push.example.invalid'
  --set 'networkPolicy.postgresEgressCidrs[0]=10.42.0.0/16'
)

helm lint "$chart" >/dev/null
helm template push "$chart" >"$default_output"
if grep -q '^kind:' "$default_output"; then
  echo "expected the default-disabled chart to render no resources" >&2
  exit 1
fi

helm lint "$chart" "${active_args[@]}" >/dev/null
helm template push "$chart" "${active_args[@]}" >"$active_output"
helm template push "$chart" "${active_args[@]}" \
  --set profiles.sandbox.enabled=true \
  --set profiles.sandbox.credentialSecretName=apns-sandbox-key \
  --set profiles.sandbox.configurationSecretName=apns-sandbox-config \
  >/dev/null
helm lint "$chart" "${production_args[@]}" >/dev/null
helm template push "$chart" "${production_args[@]}" >"$production_output"

rollback_args=(
  "${production_args[@]}"
  -f "$chart/values-rollback.yaml"
  --set rollback.targetImageDigest=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  --set-string rollback.maximumSchemaVersion=20260822000200
)
helm lint "$chart" "${rollback_args[@]}" >/dev/null
helm template push "$chart" "${rollback_args[@]}" >"$rollback_output"

ruby - "$active_output" "$production_output" "$rollback_output" <<'RUBY'
require "yaml"

active = YAML.load_stream(File.read(ARGV[0])).compact
production = YAML.load_stream(File.read(ARGV[1])).compact
rollback = YAML.load_stream(File.read(ARGV[2])).compact

deployment = active.find { |resource| resource["kind"] == "Deployment" }
service = active.find { |resource| resource["kind"] == "Service" }
job = active.find { |resource| resource["kind"] == "Job" }
raise "missing active resources" unless deployment && service && job
raise "health listener exposed by Service" unless service.dig("spec", "ports").map { |port| port["targetPort"] } == ["public"]
container = deployment.dig("spec", "template", "spec", "containers", 0)
raise "wrong liveness contract" unless container.dig("livenessProbe", "periodSeconds") == 10 && container.dig("livenessProbe", "timeoutSeconds") == 3
raise "wrong readiness contract" unless container.dig("readinessProbe", "periodSeconds") == 5 && container.dig("readinessProbe", "failureThreshold") == 3
raise "wrong startup budget" unless container.dig("startupProbe", "periodSeconds") * container.dig("startupProbe", "failureThreshold") == 120
raise "unbounded runtime resources" unless container.dig("resources", "limits", "memory") == "512Mi" && container.dig("resources", "limits", "ephemeral-storage") == "64Mi"
environment = container["env"].to_h { |entry| [entry["name"], entry] }
raise "schema version changed" unless environment.dig("ZED_PUSH_REQUIRED_SCHEMA_VERSION", "value") == "20260822000200"
raise "admission bounds changed" unless environment.dig("ZED_PUSH_PUBLIC_BODY_BYTES", "value") == "8192" && environment.dig("ZED_PUSH_PUBLIC_CONCURRENCY", "value") == "256"
raise "migration is not separately privileged" unless environment.dig("DATABASE_URL", "valueFrom", "secretKeyRef", "name") == "push-runtime" && job.dig("spec", "template", "spec", "containers", 0, "env", 2, "valueFrom", "secretKeyRef", "name") == "push-migration"
raise "migration is unbounded" unless job.dig("spec", "activeDeadlineSeconds") == 120 && job.dig("spec", "backoffLimit") == 3

runtime_labels = {
  "app.kubernetes.io/name" => "collaboration-push-gateway",
  "app.kubernetes.io/instance" => "push",
  "app.kubernetes.io/managed-by" => "Helm",
  "app.kubernetes.io/component" => "runtime"
}
migration_labels = runtime_labels.merge("app.kubernetes.io/component" => "migration")
raise "runtime selector drift" unless service.dig("spec", "selector") == runtime_labels && deployment.dig("spec", "selector", "matchLabels") == runtime_labels
raise "migration selector aliases runtime" unless job.dig("spec", "template", "metadata", "labels") == migration_labels

route = production.find { |resource| resource["kind"] == "HTTPRoute" }
raise "production route is unattached" unless route && !route.dig("spec", "parentRefs").empty?
raise "production image is mutable" unless production.find { |resource| resource["kind"] == "Deployment" }.dig("spec", "template", "spec", "containers", 0, "image").end_with?("@sha256:" + "a" * 64)

rollback_deployment = rollback.find { |resource| resource["kind"] == "Deployment" }
raise "rollback rendered a migration" if rollback.any? { |resource| resource["kind"] == "Job" || resource.dig("metadata", "name")&.end_with?("-migration") }
raise "rollback target not selected" unless rollback_deployment.dig("spec", "template", "spec", "containers", 0, "image").end_with?("@sha256:" + "b" * 64)
raise "rollback annotation missing" unless rollback_deployment.dig("spec", "template", "metadata", "annotations", "collaboration.zed.dev/rollback") == "true"
RUBY

if helm template push "$chart" "${active_args[@]}" --set runtimeSecret.name= >"$error_output" 2>&1; then
  echo "expected an empty runtime secret name to fail" >&2
  exit 1
fi
grep -q 'runtimeSecret.name is required' "$error_output"

if helm template push "$chart" "${active_args[@]}" --set profiles.production.credentialSecretName= >"$error_output" 2>&1; then
  echo "expected an empty APNs credential secret name to fail" >&2
  exit 1
fi
grep -q 'profiles.production.credentialSecretName is required' "$error_output"

if helm template push "$chart" "${active_args[@]}" \
  --set profiles.sandbox.enabled=true \
  --set profiles.sandbox.credentialSecretName=apns-production-key \
  --set profiles.sandbox.configurationSecretName=apns-sandbox-config \
  >"$error_output" 2>&1; then
  echo "expected production and sandbox APNs credentials to remain separate" >&2
  exit 1
fi
grep -q 'production and sandbox APNs credentials must use distinct secrets' "$error_output"

if helm template push "$chart" "${active_args[@]}" --set migration.secretName=push-runtime >"$error_output" 2>&1; then
  echo "expected runtime and migration credentials to remain separate" >&2
  exit 1
fi
grep -q 'runtime and DDL-capable migration credentials must use distinct secrets' "$error_output"

if helm template push "$chart" "${production_args[@]}" --set image.digest= >"$error_output" 2>&1; then
  echo "expected a production tag without a digest to fail" >&2
  exit 1
fi
grep -q 'production deployment requires image.digest' "$error_output"

if helm template push "$chart" "${production_args[@]}" -f "$chart/values-rollback.yaml" --set-string rollback.maximumSchemaVersion=20260822000200 >"$error_output" 2>&1; then
  echo "expected a rollback without a target digest to fail" >&2
  exit 1
fi
grep -q 'rollback.targetImageDigest is required' "$error_output"

if helm template push "$chart" "${production_args[@]}" -f "$chart/values-rollback.yaml" --set rollback.targetImageDigest=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb --set-string rollback.maximumSchemaVersion=20260822000199 >"$error_output" 2>&1; then
  echo "expected an incompatible rollback schema ceiling to fail" >&2
  exit 1
fi
grep -q 'rollback target does not support the deployed schema version' "$error_output"

if helm template push "$chart" "${active_args[@]}" --set limits.publicBodyBytes=8193 >"$error_output" 2>&1; then
  echo "expected an expanded public body limit to fail schema validation" >&2
  exit 1
fi
grep -q 'publicBodyBytes' "$error_output"

if helm template push "$chart" "${active_args[@]}" --set unreviewedTransport.enabled=true >"$error_output" 2>&1; then
  echo "expected an unknown configuration key to fail schema validation" >&2
  exit 1
fi
grep -q "additional properties 'unreviewedTransport' not allowed" "$error_output"
