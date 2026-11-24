import {
  ADMIN_CLIENT_ID,
  ADMIN_CLIENT_VERSION,
  COMPATIBILITY_PATH,
  AdminCollaborationError,
  communityAdminPath,
  validUuid,
  validVersion,
  type AdminCollaborationTransport,
  type AdminScope,
  type AdminTransportResponse,
  type CommunityMetricsResource,
  type CommunityResource,
  type DeletionStage,
  type DeletionStatusResource,
  type InviteResource,
  type JsonObject,
  type MemberResource,
  type ModerationReportResource,
  type WriteReceipt,
} from "./contracts.ts";

const COMMUNITY_SCOPE = "communities:manage";
const MODERATION_SCOPE = "moderation:manage";
const MAX_COLLECTION_ITEMS = 1_000;
const MAX_STRING_CHARACTERS = 4_096;

type CompatibilityFeature = "moderation" | "admin-lifecycle";
type CompatibilityAccess = "read" | "write";

export class AdminCollaborationClient {
  readonly #transport: AdminCollaborationTransport;

  constructor(transport: AdminCollaborationTransport) {
    this.#transport = transport;
  }

  async getCommunity(communityId: string): Promise<CommunityResource> {
    const path = communityAdminPath(communityId);
    return parseCommunity(await this.#resource("read", "admin-lifecycle", COMMUNITY_SCOPE, "GET", path));
  }

  async updateCommunity(
    communityId: string,
    expectedVersion: number,
    operationId: string,
    displayName: string,
  ): Promise<WriteReceipt> {
    const path = communityAdminPath(communityId);
    return parseReceipt(
      await this.#resource("write", "admin-lifecycle", COMMUNITY_SCOPE, "PATCH", path, {
        expected_version: validVersion(expectedVersion),
        operation_id: validUuid(operationId),
        display_name: validString(displayName, 1, 200),
      }),
    );
  }

  async listMembers(communityId: string): Promise<MemberResource[]> {
    const body = await this.#resource(
      "read",
      "admin-lifecycle",
      COMMUNITY_SCOPE,
      "GET",
      `${communityAdminPath(communityId)}/members`,
    );
    return parseArray(body, "members", parseMember);
  }

  async updateMemberRole(
    communityId: string,
    principalId: string,
    role: MemberResource["role"],
    expectedVersion: number,
    operationId: string,
  ): Promise<WriteReceipt> {
    const path = `${communityAdminPath(communityId)}/members/${validUuid(principalId)}`;
    return parseReceipt(
      await this.#resource("write", "admin-lifecycle", COMMUNITY_SCOPE, "PATCH", path, {
        role: enumValue(role, ["owner", "admin", "member", "guest", "bot"]),
        expected_version: validVersion(expectedVersion),
        operation_id: validUuid(operationId),
      }),
    );
  }

  async listInvites(communityId: string): Promise<InviteResource[]> {
    const body = await this.#resource(
      "read",
      "admin-lifecycle",
      COMMUNITY_SCOPE,
      "GET",
      `${communityAdminPath(communityId)}/invites`,
    );
    return parseArray(body, "invites", parseInvite);
  }

  async createInvite(
    communityId: string,
    role: InviteResource["role"],
    expiresAtMillis: number | undefined,
    maximumUses: number | undefined,
    operationId: string,
  ): Promise<WriteReceipt> {
    return parseReceipt(
      await this.#resource(
        "write",
        "admin-lifecycle",
        COMMUNITY_SCOPE,
        "POST",
        `${communityAdminPath(communityId)}/invites`,
        {
          role: enumValue(role, ["member", "guest"]),
          expires_at_millis: optionalSafeInteger(expiresAtMillis, 1),
          maximum_uses: optionalSafeInteger(maximumUses, 1, 10_000),
          operation_id: validUuid(operationId),
        },
      ),
    );
  }

  async revokeInvite(
    communityId: string,
    inviteId: string,
    expectedVersion: number,
    operationId: string,
  ): Promise<WriteReceipt> {
    return parseReceipt(
      await this.#resource(
        "write",
        "admin-lifecycle",
        COMMUNITY_SCOPE,
        "POST",
        `${communityAdminPath(communityId)}/invites/${validUuid(inviteId)}/revoke`,
        {
          expected_version: validVersion(expectedVersion),
          operation_id: validUuid(operationId),
        },
      ),
    );
  }

  async listModerationReports(communityId: string): Promise<ModerationReportResource[]> {
    const body = await this.#resource(
      "read",
      "moderation",
      MODERATION_SCOPE,
      "GET",
      `${communityAdminPath(communityId)}/moderation/reports`,
    );
    return parseArray(body, "reports", parseReport);
  }

  async resolveModerationReport(
    communityId: string,
    reportId: string,
    expectedVersion: number,
    resolution: "dismissed" | "actioned",
    operationId: string,
  ): Promise<WriteReceipt> {
    return parseReceipt(
      await this.#resource(
        "write",
        "moderation",
        MODERATION_SCOPE,
        "POST",
        `${communityAdminPath(communityId)}/moderation/reports/${validUuid(reportId)}/resolve`,
        {
          expected_version: validVersion(expectedVersion),
          resolution: enumValue(resolution, ["dismissed", "actioned"]),
          operation_id: validUuid(operationId),
        },
      ),
    );
  }

  async archiveCommunity(
    communityId: string,
    expectedVersion: number | undefined,
    operationId: string,
  ): Promise<WriteReceipt> {
    return parseReceipt(
      await this.#resource(
        "write",
        "admin-lifecycle",
        COMMUNITY_SCOPE,
        "POST",
        `${communityAdminPath(communityId)}/archive`,
        {
          expected_version: expectedVersion === undefined ? undefined : validVersion(expectedVersion),
          operation_id: validUuid(operationId),
        },
      ),
    );
  }

  async getDeletionStatus(communityId: string, deletionId: string): Promise<DeletionStatusResource> {
    const body = await this.#resource(
      "read",
      "admin-lifecycle",
      COMMUNITY_SCOPE,
      "GET",
      `${communityAdminPath(communityId)}/deletions/${validUuid(deletionId)}`,
    );
    return parseDeletionStatus(body);
  }

  async restoreDeletion(
    communityId: string,
    deletionId: string,
    expectedVersion: number,
    operationId: string,
  ): Promise<DeletionStatusResource> {
    const body = await this.#resource(
      "write",
      "admin-lifecycle",
      COMMUNITY_SCOPE,
      "POST",
      `${communityAdminPath(communityId)}/deletions/${validUuid(deletionId)}/restore`,
      {
        expected_version: validVersion(expectedVersion),
        operation_id: validUuid(operationId),
      },
    );
    return parseDeletionStatus(body);
  }

  async getMetrics(communityId: string): Promise<CommunityMetricsResource> {
    return parseMetrics(
      await this.#resource(
        "read",
        "admin-lifecycle",
        COMMUNITY_SCOPE,
        "GET",
        `${communityAdminPath(communityId)}/metrics`,
      ),
    );
  }

  async #resource(
    access: CompatibilityAccess,
    feature: CompatibilityFeature,
    requiredScope: AdminScope,
    method: "GET" | "POST" | "PATCH",
    path: string,
    body?: JsonObject,
  ): Promise<JsonObject> {
    await this.#negotiate(access, feature);
    let response: AdminTransportResponse;
    try {
      response = await this.#transport.request({
        method,
        path,
        requiredScope,
        body,
      });
    } catch (error) {
      if (error instanceof AdminCollaborationError) throw error;
      throw serviceUnavailable();
    }
    if (response.status < 200 || response.status >= 300) {
      throw resourceError(response);
    }
    return jsonObject(response.body);
  }

  async #negotiate(access: CompatibilityAccess, feature: CompatibilityFeature): Promise<void> {
    let response: AdminTransportResponse;
    try {
      response = await this.#transport.request({
        method: "POST",
        path: COMPATIBILITY_PATH,
        body: {
          client_id: ADMIN_CLIENT_ID,
          client_version: ADMIN_CLIENT_VERSION,
          access,
          protocols: [{ id: "collaboration-http", version: 1 }],
          features: [feature],
        },
      });
    } catch (error) {
      if (error instanceof AdminCollaborationError) throw error;
      throw serviceUnavailable();
    }
    const body = jsonObject(response.body);
    if (response.status === 426 || body.error === "upgrade_required") {
      const minimumVersion = optionalString(body.minimum_client_version);
      const maximumVersion = optionalString(body.maximum_client_version);
      throw new AdminCollaborationError("upgrade_required", versionMessage(minimumVersion, maximumVersion), {
        minimumVersion,
        maximumVersion,
      });
    }
    if (response.status < 200 || response.status >= 300) {
      throw serviceUnavailable();
    }
    const features = body.selected_features;
    if (
      !Number.isInteger(body.policy_version) ||
      (body.policy_version as number) < 1 ||
      body.outcome !== "supported" ||
      body.client_id !== ADMIN_CLIENT_ID ||
      body.retryable !== false ||
      !Array.isArray(features) ||
      features.length !== 1 ||
      features[0] !== feature
    ) {
      throw new AdminCollaborationError("invalid_response", "The administration service returned an invalid response.");
    }
  }
}

function parseCommunity(value: JsonObject): CommunityResource {
  return {
    communityId: validUuid(requiredString(value.community_id)),
    displayName: validString(requiredString(value.display_name), 1, 200),
    state: enumValue(value.state, ["active", "archived", "deleting", "deleted"]),
    version: responseVersion(value.version),
  };
}

function parseMember(value: unknown): MemberResource {
  const body = jsonObject(value);
  return {
    principalId: validUuid(requiredString(body.principal_id)),
    role: enumValue(body.role, ["owner", "admin", "member", "guest", "bot"]),
    state: enumValue(body.state, ["active", "archived"]),
    version: responseVersion(body.version),
  };
}

function parseInvite(value: unknown): InviteResource {
  const body = jsonObject(value);
  return {
    inviteId: validUuid(requiredString(body.invite_id)),
    role: enumValue(body.role, ["member", "guest"]),
    state: enumValue(body.state, ["active", "expired", "exhausted", "revoked"]),
    expiresAtMillis: responseOptionalInteger(body.expires_at_millis),
    remainingUses: responseOptionalInteger(body.remaining_uses),
    version: responseVersion(body.version),
  };
}

function parseReport(value: unknown): ModerationReportResource {
  const body = jsonObject(value);
  return {
    reportId: validUuid(requiredString(body.report_id)),
    targetKind: enumValue(body.target_kind, ["event", "principal", "blob"]),
    targetId: validString(requiredString(body.target_id), 1, 512),
    reason: validString(requiredString(body.reason), 1, 1_000),
    state: enumValue(body.state, ["open", "resolved"]),
    version: responseVersion(body.version),
  };
}

function parseReceipt(value: JsonObject): WriteReceipt {
  return {
    operationId: validUuid(requiredString(value.operation_id)),
    resourceId: validUuid(requiredString(value.resource_id)),
    version: responseVersion(value.version),
  };
}

function parseDeletionStatus(value: JsonObject): DeletionStatusResource {
  const stages: DeletionStage[] = [
    "requested",
    "verified",
    "reversible",
    "irreversible",
    "failed",
    "deleted",
    "rolled_back",
  ];
  const totalPhases = responseInteger(value.total_phases, 1, 6);
  const completedPhases = responseInteger(value.completed_phases, 0, totalPhases);
  return {
    stage: enumValue(value.stage, stages),
    lastTrustworthyStage: enumValue(value.last_trustworthy_stage, stages),
    completedPhases,
    totalPhases,
    nextPhase: optionalEnum(value.next_phase, ["database", "search", "cache", "push", "object_storage", "git"]),
    checkpointVersion: responseOptionalInteger(value.checkpoint_version),
    haltReason: optionalEnum(value.halt_reason, [
      "authority_unavailable",
      "inventory_mismatch",
      "dependency_unavailable",
      "fence_lost",
      "verification_failed",
      "execution_conflict",
    ]),
    recoveryAction: enumValue(value.recovery_action, ["none", "restore", "resume"]),
  };
}

function parseMetrics(value: JsonObject): CommunityMetricsResource {
  return {
    activeMembers: responseInteger(value.active_members, 0),
    openReports: responseInteger(value.open_reports, 0),
    pendingInvites: responseInteger(value.pending_invites, 0),
    deletionCompletedPhases: responseInteger(value.deletion_completed_phases, 0, 6),
    deletionTotalPhases: responseInteger(value.deletion_total_phases, 0, 6),
    measuredAtMillis: responseInteger(value.measured_at_millis, 1),
  };
}

function parseArray<T>(body: JsonObject, field: string, parse: (value: unknown) => T): T[] {
  const values = body[field];
  if (!Array.isArray(values) || values.length > MAX_COLLECTION_ITEMS) {
    throw invalidResponse();
  }
  return values.map(parse);
}

function jsonObject(value: unknown): JsonObject {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw invalidResponse();
  }
  return value as JsonObject;
}

function requiredString(value: unknown): string {
  if (typeof value !== "string") throw invalidResponse();
  return value;
}

function validString(value: string, minimum: number, maximum: number): string {
  if (
    value.length < minimum ||
    value.length > Math.min(maximum, MAX_STRING_CHARACTERS) ||
    Array.from(value).some((character) => {
      const codePoint = character.codePointAt(0);
      return codePoint !== undefined && (codePoint <= 31 || codePoint === 127);
    })
  ) {
    throw new AdminCollaborationError("invalid_request", "The administration request is invalid.");
  }
  return value;
}

function enumValue<const T extends string>(value: unknown, values: readonly T[]): T {
  if (typeof value !== "string" || !values.includes(value as T)) {
    throw invalidResponse();
  }
  return value as T;
}

function optionalEnum<const T extends string>(value: unknown, values: readonly T[]): T | undefined {
  return value === undefined || value === null ? undefined : enumValue(value, values);
}

function responseVersion(value: unknown): number {
  return responseInteger(value, 1);
}

function responseOptionalInteger(value: unknown): number | undefined {
  return value === undefined || value === null ? undefined : responseInteger(value, 0);
}

function responseInteger(value: unknown, minimum: number, maximum?: number): number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < minimum ||
    (maximum !== undefined && (value as number) > maximum)
  ) {
    throw invalidResponse();
  }
  return value as number;
}

function optionalSafeInteger(value: number | undefined, minimum: number, maximum?: number): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || value < minimum || (maximum !== undefined && value > maximum)) {
    throw new AdminCollaborationError("invalid_request", "The administration request is invalid.");
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length <= 64 ? value : undefined;
}

function resourceError(response: AdminTransportResponse): AdminCollaborationError {
  const body =
    response.body !== null && !Array.isArray(response.body) && typeof response.body === "object"
      ? (response.body as JsonObject)
      : {};
  const error = body.error;
  if (response.status === 401 || response.status === 403) {
    return new AdminCollaborationError("authorization_denied", "The administration request was denied.");
  }
  if (response.status === 404) {
    return new AdminCollaborationError("resource_unavailable", "The administration resource is unavailable.");
  }
  if (response.status === 409 || error === "stale_write") {
    return new AdminCollaborationError("stale_write", "The resource changed. Reload it before retrying.");
  }
  if (error === "outcome_unknown") {
    return new AdminCollaborationError(
      "outcome_unknown",
      "The operation outcome is unknown. Reload the resource before retrying.",
      { retryable: true },
    );
  }
  if (response.status >= 500) return serviceUnavailable();
  return new AdminCollaborationError("invalid_request", "The administration request is invalid.");
}

function invalidResponse(): AdminCollaborationError {
  return new AdminCollaborationError("invalid_response", "The administration service returned an invalid response.");
}

function serviceUnavailable(): AdminCollaborationError {
  return new AdminCollaborationError("service_unavailable", "The administration service is unavailable.", {
    retryable: true,
  });
}

function versionMessage(minimum?: string, maximum?: string): string {
  if (minimum !== undefined && maximum !== undefined) {
    return `Upgrade the administration client to a supported version (${minimum} through ${maximum}).`;
  }
  return "Upgrade the administration client before continuing.";
}
