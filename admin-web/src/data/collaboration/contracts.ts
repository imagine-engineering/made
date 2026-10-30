export const ADMIN_CLIENT_ID = "buzz-admin-web";
export const ADMIN_CLIENT_VERSION = "0.1.0";
export const COMPATIBILITY_PATH = "/v1/collaboration/compatibility";
export const ADMIN_API_PATH = "/v1/collaboration/admin";

export type AdminScope = "communities:manage" | "moderation:manage";
export type AdminMethod = "GET" | "POST" | "PATCH";
export type JsonObject = Record<string, unknown>;

export type AdminTransportRequest = {
  method: AdminMethod;
  path: string;
  requiredScope?: AdminScope;
  body?: JsonObject;
};

export type AdminTransportResponse = {
  status: number;
  body: unknown;
};

export interface AdminCollaborationTransport {
  request(request: AdminTransportRequest): Promise<AdminTransportResponse>;
}

export type CommunityResource = {
  communityId: string;
  displayName: string;
  state: "active" | "archived" | "deleting" | "deleted";
  version: number;
};

export type MemberResource = {
  principalId: string;
  role: "owner" | "admin" | "member" | "guest" | "bot";
  state: "active" | "archived";
  version: number;
};

export type InviteResource = {
  inviteId: string;
  role: "member" | "guest";
  state: "active" | "expired" | "exhausted" | "revoked";
  expiresAtMillis?: number;
  remainingUses?: number;
  version: number;
};

export type ModerationReportResource = {
  reportId: string;
  targetKind: "event" | "principal" | "blob";
  targetId: string;
  reason: string;
  state: "open" | "resolved";
  version: number;
};

export type WriteReceipt = {
  operationId: string;
  resourceId: string;
  version: number;
};

export type DeletionStage =
  | "requested"
  | "verified"
  | "reversible"
  | "irreversible"
  | "failed"
  | "deleted"
  | "rolled_back";

export type DeletionStatusResource = {
  stage: DeletionStage;
  lastTrustworthyStage: DeletionStage;
  completedPhases: number;
  totalPhases: number;
  nextPhase?: "database" | "search" | "cache" | "push" | "object_storage" | "git";
  checkpointVersion?: number;
  haltReason?:
    | "authority_unavailable"
    | "inventory_mismatch"
    | "dependency_unavailable"
    | "fence_lost"
    | "verification_failed"
    | "execution_conflict";
  recoveryAction: "none" | "restore" | "resume";
};

export type CommunityMetricsResource = {
  activeMembers: number;
  openReports: number;
  pendingInvites: number;
  deletionCompletedPhases: number;
  deletionTotalPhases: number;
  measuredAtMillis: number;
};

export type AdminCollaborationErrorKind =
  | "upgrade_required"
  | "authorization_denied"
  | "resource_unavailable"
  | "stale_write"
  | "outcome_unknown"
  | "invalid_request"
  | "invalid_response"
  | "service_unavailable";

export class AdminCollaborationError extends Error {
  readonly kind: AdminCollaborationErrorKind;
  readonly minimumVersion?: string;
  readonly maximumVersion?: string;
  readonly retryable: boolean;

  constructor(
    kind: AdminCollaborationErrorKind,
    message: string,
    options?: {
      minimumVersion?: string;
      maximumVersion?: string;
      retryable?: boolean;
    },
  ) {
    super(message);
    this.name = "AdminCollaborationError";
    this.kind = kind;
    this.minimumVersion = options?.minimumVersion;
    this.maximumVersion = options?.maximumVersion;
    this.retryable = options?.retryable ?? false;
  }
}

export function communityAdminPath(communityId: string): string {
  return `${ADMIN_API_PATH}/communities/${validUuid(communityId)}`;
}

export function validUuid(value: string): string {
  if (
    !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value) ||
    value === "00000000-0000-0000-0000-000000000000"
  ) {
    throw new AdminCollaborationError("invalid_request", "The administration request is invalid.");
  }
  return value.toLowerCase();
}

export function validVersion(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new AdminCollaborationError("invalid_request", "The administration request is invalid.");
  }
  return value;
}
