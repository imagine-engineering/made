export const WEB_CLIENT_ID = "buzz-web";
export const WEB_CLIENT_VERSION = "0.1.0";
export const MINIMUM_COMPATIBILITY_POLICY_VERSION = 1;
export const COMPATIBILITY_PATH = "/v1/collaboration/compatibility";
export const INVITE_RESOLVE_PATH = "/v1/collaboration/invites/resolve";
export const INVITE_REDEEM_PATH = "/v1/collaboration/invites/redeem";
export const INVITE_ROUTE = "/invite/$code";

export type CompatibilityAccess = "read" | "write";

export type CompatibilityResponse = {
  policy_version: number;
  outcome: "supported" | "read_only" | "upgrade_required";
  error?: string;
  reason?: string;
  client_id: string;
  minimum_client_version?: string;
  maximum_client_version?: string;
  selected_features: string[];
  retryable: boolean;
};

export type InviteStatus = "active" | "expired" | "exhausted" | "revoked";

export type JoinPolicy = {
  version: string;
  terms_markdown?: string;
  privacy_markdown?: string;
  age_attestation_required: boolean;
};

export type ResolvedInvite = {
  communityId: string;
  communityHost: string;
  status: "active";
  role: "member" | "guest";
  joinPolicy: JoinPolicy | null;
};

export type InvitePolicyAcceptance = {
  policyVersion: string;
  ageConfirmed: boolean;
  legalDocumentsAccepted: boolean;
};

export type InviteRedemption = {
  status: "joined" | "already_member";
  communityId: string;
  communityHost: string;
  role: "member" | "guest";
};

export type CollaborationAuthErrorKind =
  | "upgrade_required"
  | "read_only"
  | "invite_expired"
  | "invite_exhausted"
  | "invite_revoked"
  | "invite_invalid"
  | "signer_denied"
  | "signer_invalid"
  | "service_unavailable"
  | "invalid_response";

export class CollaborationAuthError extends Error {
  readonly kind: CollaborationAuthErrorKind;
  readonly minimumVersion?: string;
  readonly maximumVersion?: string;
  readonly retryable: boolean;

  constructor(
    kind: CollaborationAuthErrorKind,
    message: string,
    options?: {
      minimumVersion?: string;
      maximumVersion?: string;
      retryable?: boolean;
    },
  ) {
    super(message);
    this.name = "CollaborationAuthError";
    this.kind = kind;
    this.minimumVersion = options?.minimumVersion;
    this.maximumVersion = options?.maximumVersion;
    this.retryable = options?.retryable ?? false;
  }
}

export function invitePagePath(code: string): string {
  return `/invite/${encodeURIComponent(validInviteCode(code))}`;
}

export function buzzInviteDeepLink(
  relayUrl: string,
  code: string,
  policyReceipt?: string,
): string {
  let relay: URL;
  try {
    relay = new URL(relayUrl);
  } catch {
    throw invalidRelay();
  }
  if (
    !["ws:", "wss:"].includes(relay.protocol) ||
    relay.username ||
    relay.password ||
    relay.search ||
    relay.hash
  ) {
    throw invalidRelay();
  }
  const query = new URLSearchParams({
    relay: relayUrl,
    code: validInviteCode(code),
  });
  if (policyReceipt !== undefined) {
    if (
      policyReceipt.length === 0 ||
      policyReceipt.length > 2_048 ||
      containsControlCharacter(policyReceipt)
    ) {
      throw new CollaborationAuthError(
        "invite_invalid",
        "The invite policy receipt is invalid. Request a new invite.",
      );
    }
    query.set("policy_receipt", policyReceipt);
  }
  return `buzz://join?${query.toString()}`;
}

function invalidRelay(): CollaborationAuthError {
  return new CollaborationAuthError(
    "invite_invalid",
    "The invite relay is invalid. Check the link or ask for a new invite.",
  );
}

export function communityJoinPolicyPath(communityId: string): string {
  if (
    !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(communityId) ||
    communityId === "00000000-0000-0000-0000-000000000000"
  ) {
    throw new CollaborationAuthError(
      "invalid_response",
      "The collaboration service returned an invalid community identity.",
    );
  }
  return `/v1/collaboration/communities/${communityId.toLowerCase()}/join-policy`;
}

export function validInviteCode(code: string): string {
  if (
    code.length === 0 ||
    code.length > 256 ||
    containsControlCharacter(code) ||
    /\s/u.test(code)
  ) {
    throw new CollaborationAuthError(
      "invite_invalid",
      "This invite link is invalid. Check the link or ask for a new invite.",
    );
  }
  return code;
}

function containsControlCharacter(value: string): boolean {
  return Array.from(value).some((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint !== undefined && (codePoint <= 31 || codePoint === 127);
  });
}
