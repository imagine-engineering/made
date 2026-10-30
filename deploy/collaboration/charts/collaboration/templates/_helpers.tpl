{{- define "collaboration.name" -}}
{{- printf "%s-collaboration" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "collaboration.labels" -}}
app.kubernetes.io/name: collaboration
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: zed
{{- end -}}

{{- define "collaboration.runtimeLabels" -}}
{{ include "collaboration.labels" . }}
app.kubernetes.io/component: service
{{- end -}}

{{- define "collaboration.migrationLabels" -}}
{{ include "collaboration.labels" . }}
app.kubernetes.io/component: migration
{{- end -}}

{{- define "collaboration.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "collaboration.name" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "collaboration.maximumReplicas" -}}
{{- if .Values.autoscaling.enabled -}}
{{- .Values.autoscaling.maxReplicas -}}
{{- else -}}
{{- .Values.deployment.replicaCount -}}
{{- end -}}
{{- end -}}

{{- define "collaboration.minimumReplicas" -}}
{{- if .Values.autoscaling.enabled -}}
{{- .Values.autoscaling.minReplicas -}}
{{- else -}}
{{- .Values.deployment.replicaCount -}}
{{- end -}}
{{- end -}}

{{- define "collaboration.image" -}}
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

{{- define "collaboration.migrationImage" -}}
{{- if .Values.deployment.production -}}
{{ printf "%s@%s" .Values.migration.image.repository (required "production migration requires migration.image.digest" .Values.migration.image.digest) }}
{{- else if .Values.migration.image.digest -}}
{{ printf "%s@%s" .Values.migration.image.repository .Values.migration.image.digest }}
{{- else -}}
{{ printf "%s:%s" .Values.migration.image.repository (required "migration.image.tag or migration.image.digest is required" .Values.migration.image.tag) }}
{{- end -}}
{{- end -}}

{{- define "collaboration.validate" -}}
{{- if .Values.deployment.enabled -}}
{{- $_ := required "runtimeSecret.name is required when deployment.enabled=true" .Values.runtimeSecret.name -}}
{{- $_ := required "publicUrl is required when deployment.enabled=true" .Values.publicUrl -}}
{{- $_ := required "objectStore.endpoint is required when deployment.enabled=true" .Values.objectStore.endpoint -}}
{{- if and .Values.deployment.production (not (regexMatch "^https://" .Values.publicUrl)) -}}
{{- fail "production publicUrl must use https" -}}
{{- end -}}
{{- if and .Values.deployment.production (not (regexMatch "^https://" .Values.objectStore.endpoint)) -}}
{{- fail "production objectStore.endpoint must use https" -}}
{{- end -}}
{{- if and .Values.ingress.enabled .Values.httpRoute.enabled -}}
{{- fail "ingress.enabled and httpRoute.enabled cannot both be true" -}}
{{- end -}}
{{- if and .Values.deployment.production (not (or .Values.ingress.enabled .Values.httpRoute.enabled)) -}}
{{- fail "production deployment requires ingress.enabled or httpRoute.enabled" -}}
{{- end -}}
{{- if .Values.ingress.enabled -}}
{{- $_ := required "ingress.hostname is required when ingress.enabled=true" .Values.ingress.hostname -}}
{{- $_ := required "ingress.tlsSecretName is required when ingress.enabled=true" .Values.ingress.tlsSecretName -}}
{{- end -}}
{{- if .Values.httpRoute.enabled -}}
{{- if not .Values.httpRoute.parentRefs -}}
{{- fail "httpRoute.enabled requires an explicit parentRefs attachment" -}}
{{- end -}}
{{- if not .Values.httpRoute.hostnames -}}
{{- fail "httpRoute.enabled requires at least one hostname" -}}
{{- end -}}
{{- end -}}
{{- if .Values.nostrHttp.enabled -}}
{{- if not .Values.redis.enabled -}}
{{- fail "nostrHttp.enabled requires redis.enabled=true" -}}
{{- end -}}
{{- $_ := required "Nostr HTTP requires runtimeSecret.redisUrlKey" .Values.runtimeSecret.redisUrlKey -}}
{{- if not .Values.nostrHttp.routes -}}
{{- fail "nostrHttp.enabled requires explicit community routes" -}}
{{- end -}}
{{- if eq (toJson .Values.nostrHttp.callsPerMinute) "null" -}}
{{- fail "nostrHttp.enabled requires an explicit callsPerMinute quota" -}}
{{- end -}}
{{- end -}}
{{- if and .Values.autoscaling.enabled (lt (int .Values.autoscaling.maxReplicas) (int .Values.autoscaling.minReplicas)) -}}
{{- fail "autoscaling.maxReplicas must be greater than or equal to autoscaling.minReplicas" -}}
{{- end -}}
{{- if gt (include "collaboration.maximumReplicas" . | int) 1 -}}
{{- if not .Values.redis.enabled -}}
{{- fail "multiple replicas require redis.enabled=true" -}}
{{- end -}}
{{- $_ := required "multiple replicas require runtimeSecret.gitHookSecretKey" .Values.runtimeSecret.gitHookSecretKey -}}
{{- end -}}
{{- if and .Values.deployment.production (not .Values.persistence.git.enabled) -}}
{{- fail "production deployment requires persistent Git storage" -}}
{{- end -}}
{{- if and .Values.persistence.git.enabled (gt (include "collaboration.maximumReplicas" . | int) 1) (ne .Values.persistence.git.accessMode "ReadWriteMany") -}}
{{- fail "multiple replicas with persistent Git storage require ReadWriteMany" -}}
{{- end -}}
{{- if ge (int .Values.podDisruptionBudget.minAvailable) (include "collaboration.minimumReplicas" . | int) -}}
{{- fail "podDisruptionBudget.minAvailable must be lower than the minimum replica count" -}}
{{- end -}}
{{- if not .Values.networkPolicy.postgresEgressCidrs -}}
{{- fail "networkPolicy.postgresEgressCidrs must name the database network" -}}
{{- end -}}
{{- if not .Values.networkPolicy.objectStoreEgressCidrs -}}
{{- fail "networkPolicy.objectStoreEgressCidrs must name the object-store network" -}}
{{- end -}}
{{- if and .Values.redis.enabled (not .Values.networkPolicy.redisEgressCidrs) -}}
{{- fail "redis.enabled requires networkPolicy.redisEgressCidrs" -}}
{{- end -}}
{{- if .Values.push.enabled -}}
{{- $_ := required "push.url is required when push.enabled=true" .Values.push.url -}}
{{- if not .Values.networkPolicy.optionalHttpsEgressCidrs -}}
{{- fail "push requires networkPolicy.optionalHttpsEgressCidrs" -}}
{{- end -}}
{{- end -}}
{{- if .Values.pairing.enabled -}}
{{- $_ := required "pairing.url is required when pairing.enabled=true" .Values.pairing.url -}}
{{- if not .Values.networkPolicy.optionalHttpsEgressCidrs -}}
{{- fail "pairing requires networkPolicy.optionalHttpsEgressCidrs" -}}
{{- end -}}
{{- end -}}
{{- if .Values.relayMesh.enabled -}}
{{- if lt (include "collaboration.minimumReplicas" . | int) 2 -}}
{{- fail "relayMesh requires at least two replicas" -}}
{{- end -}}
{{- if not .Values.redis.enabled -}}
{{- fail "relayMesh requires redis.enabled=true" -}}
{{- end -}}
{{- if not .Values.relayMesh.peers -}}
{{- fail "relayMesh.peers is required when relayMesh.enabled=true" -}}
{{- end -}}
{{- if not .Values.networkPolicy.optionalHttpsEgressCidrs -}}
{{- fail "relayMesh requires networkPolicy.optionalHttpsEgressCidrs" -}}
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
{{- if and .Values.migration.enabled (not .Values.rollback.enabled) -}}
{{- $_ := required "migration.secretName is required when migrations are enabled" .Values.migration.secretName -}}
{{- if eq .Values.runtimeSecret.name .Values.migration.secretName -}}
{{- fail "runtime and DDL-capable migration credentials must use distinct secrets" -}}
{{- end -}}
{{- if .Values.deployment.production -}}
{{- $_ := required "production migration requires migration.image.digest" .Values.migration.image.digest -}}
{{- else if not (or .Values.migration.image.tag .Values.migration.image.digest) -}}
{{- fail "migration.image.tag or migration.image.digest is required" -}}
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

{{- define "collaboration.nostrHttpRoutes" -}}
{{- $routes := list -}}
{{- range .Values.nostrHttp.routes -}}
{{- $routes = append $routes (dict "community_id" .communityId "public_url" .publicUrl) -}}
{{- end -}}
{{- toJson $routes -}}
{{- end -}}
