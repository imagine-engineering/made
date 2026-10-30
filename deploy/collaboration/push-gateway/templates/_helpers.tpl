{{- define "collaborationPush.name" -}}
{{- printf "%s-collaboration-push-gateway" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "collaborationPush.labels" -}}
app.kubernetes.io/name: collaboration-push-gateway
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "collaborationPush.runtimeLabels" -}}
{{ include "collaborationPush.labels" . }}
app.kubernetes.io/component: runtime
{{- end -}}

{{- define "collaborationPush.migrationLabels" -}}
{{ include "collaborationPush.labels" . }}
app.kubernetes.io/component: migration
{{- end -}}

{{- define "collaborationPush.image" -}}
{{- if .Values.rollback.enabled -}}
{{ printf "%s@%s" .Values.image.repository (required "rollback.targetImageDigest is required in rollback mode" .Values.rollback.targetImageDigest) }}
{{- else if .Values.deployment.production -}}
{{ printf "%s@%s" .Values.image.repository (required "production deployment requires image.digest" .Values.image.digest) }}
{{- else if .Values.image.digest -}}
{{ printf "%s@%s" .Values.image.repository .Values.image.digest }}
{{- else -}}
{{ printf "%s:%s" .Values.image.repository (required "image.tag or image.digest is required" .Values.image.tag) }}
{{- end -}}
{{- end -}}

{{- define "collaborationPush.validateProfile" -}}
{{- $profile := index . 0 -}}
{{- $name := index . 1 -}}
{{- if $profile.enabled -}}
{{- $_ := required (printf "profiles.%s.credentialSecretName is required" $name) $profile.credentialSecretName -}}
{{- $_ := required (printf "profiles.%s.configurationSecretName is required" $name) $profile.configurationSecretName -}}
{{- end -}}
{{- end -}}

{{- define "collaborationPush.validate" -}}
{{- if .Values.deployment.enabled -}}
{{- $_ := required "runtimeSecret.name is required when deployment.enabled=true" .Values.runtimeSecret.name -}}
{{- $_ := required "publicDeliveryUrl is required when deployment.enabled=true" .Values.publicDeliveryUrl -}}
{{- $_ := required "appAttest.appIdentifier is required when deployment.enabled=true" .Values.appAttest.appIdentifier -}}
{{- $_ := required "appAttest.rootCertificateSecretName is required when deployment.enabled=true" .Values.appAttest.rootCertificateSecretName -}}
{{- if not (regexMatch "^https://[^/?#]+/v1/deliveries/apns$" .Values.publicDeliveryUrl) -}}
{{- fail "publicDeliveryUrl must be an https origin plus /v1/deliveries/apns" -}}
{{- end -}}
{{- if eq .Values.runtimeSecret.grantKeysKey .Values.runtimeSecret.tokenKeysKey -}}
{{- fail "runtimeSecret grant and token key entries must be distinct" -}}
{{- end -}}
{{- if ne .Values.profiles.production.identifier "buzz-ios-production" -}}
{{- fail "profiles.production.identifier must remain buzz-ios-production" -}}
{{- end -}}
{{- if ne .Values.profiles.sandbox.identifier "buzz-ios-sandbox" -}}
{{- fail "profiles.sandbox.identifier must remain buzz-ios-sandbox" -}}
{{- end -}}
{{- if not (or .Values.profiles.production.enabled .Values.profiles.sandbox.enabled) -}}
{{- fail "at least one approved APNs profile must be enabled" -}}
{{- end -}}
{{- include "collaborationPush.validateProfile" (list .Values.profiles.production "production") -}}
{{- include "collaborationPush.validateProfile" (list .Values.profiles.sandbox "sandbox") -}}
{{- if and .Values.profiles.production.enabled .Values.profiles.sandbox.enabled -}}
{{- if eq .Values.profiles.production.credentialSecretName .Values.profiles.sandbox.credentialSecretName -}}
{{- fail "production and sandbox APNs credentials must use distinct secrets" -}}
{{- end -}}
{{- if eq .Values.profiles.production.configurationSecretName .Values.profiles.sandbox.configurationSecretName -}}
{{- fail "production and sandbox APNs configuration must use distinct secrets" -}}
{{- end -}}
{{- end -}}
{{- if .Values.migration.enabled -}}
{{- $_ := required "migration.secretName is required when migrations are enabled" .Values.migration.secretName -}}
{{- if eq .Values.runtimeSecret.name .Values.migration.secretName -}}
{{- fail "runtime and DDL-capable migration credentials must use distinct secrets" -}}
{{- end -}}
{{- end -}}
{{- if not .Values.networkPolicy.postgresEgressCidrs -}}
{{- fail "networkPolicy.postgresEgressCidrs must name the database network" -}}
{{- end -}}
{{- if .Values.httpRoute.enabled -}}
{{- if not .Values.httpRoute.parentRefs -}}
{{- fail "httpRoute.enabled requires an explicit parentRefs attachment" -}}
{{- end -}}
{{- if not .Values.httpRoute.hostnames -}}
{{- fail "httpRoute.enabled requires at least one hostname" -}}
{{- end -}}
{{- end -}}
{{- if ne .Values.podMonitor.enabled .Values.networkPolicy.monitoring.enabled -}}
{{- fail "podMonitor and scoped monitoring ingress must be enabled together" -}}
{{- end -}}
{{- if .Values.networkPolicy.monitoring.enabled -}}
{{- if or (not .Values.networkPolicy.monitoring.namespaceSelector) (not .Values.networkPolicy.monitoring.podSelector) -}}
{{- fail "monitoring ingress requires non-empty namespace and pod selectors" -}}
{{- end -}}
{{- end -}}
{{- if .Values.rollback.enabled -}}
{{- $_ := required "rollback.targetImageDigest is required in rollback mode" .Values.rollback.targetImageDigest -}}
{{- if lt (int64 .Values.rollback.maximumSchemaVersion) (int64 .Values.migration.requiredSchemaVersion) -}}
{{- fail "rollback target does not support the deployed schema version" -}}
{{- end -}}
{{- else if .Values.deployment.production -}}
{{- $_ := required "production deployment requires image.digest" .Values.image.digest -}}
{{- end -}}
{{- end -}}
{{- end -}}
