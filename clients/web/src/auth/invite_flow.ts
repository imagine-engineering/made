import {
  COMPATIBILITY_PATH,
  CollaborationAuthError,
  INVITE_REDEEM_PATH,
  INVITE_RESOLVE_PATH,
  MINIMUM_COMPATIBILITY_POLICY_VERSION,
  WEB_CLIENT_ID,
  WEB_CLIENT_VERSION,
  communityJoinPolicyPath,
  type CompatibilityAccess,
  type CompatibilityResponse,
  type InvitePolicyAcceptance,
  type InviteRedemption,
  type InviteStatus,
  type JoinPolicy,
  type ResolvedInvite,
  validInviteCode,
} from "./contracts.ts";
import { makeNip98Authorization, type Nip07Provider } from "./nip98.ts";

const REQUEST_TIMEOUT_MS = 15_000;
const MAX_RESPONSE_CHARACTERS = 64 * 1_024;
const REQUIRED_FEATURES = ["communities", "invites"] as const;

type JsonObject = Record<string, unknown>;

export type CollaborationWebAuthClientOptions = {
  baseUrl: string;
  fetch?: typeof fetch;
  signer?: Nip07Provider;
  timeoutMilliseconds?: number;
};

export class CollaborationWebAuthClient {
  readonly #baseUrl: URL;
  readonly #fetch: typeof fetch;
  readonly #signer: Nip07Provider | undefined;
  readonly #timeoutMilliseconds: number;

  constructor(options: CollaborationWebAuthClientOptions) {
    let baseUrl: URL;
    try {
      baseUrl = new URL(options.baseUrl);
    } catch {
      throw invalidServiceUrl();
    }
    const localHttp =
      baseUrl.protocol === "http:" &&
      (baseUrl.hostname === "localhost" ||
        baseUrl.hostname.endsWith(".localhost") ||
        baseUrl.hostname === "127.0.0.1" ||
        baseUrl.hostname === "[::1]");
    if (
      (baseUrl.protocol !== "https:" && !localHttp) ||
      baseUrl.username ||
      baseUrl.password ||
      baseUrl.search ||
      baseUrl.hash
    ) {
      throw invalidServiceUrl();
    }
    const timeoutMilliseconds =
      options.timeoutMilliseconds ?? REQUEST_TIMEOUT_MS;
    if (
      !Number.isInteger(timeoutMilliseconds) ||
      timeoutMilliseconds < 1 ||
      timeoutMilliseconds > 60_000
    ) {
      throw invalidServiceUrl();
    }
    baseUrl.pathname = "/";
    this.#baseUrl = baseUrl;
    this.#fetch = options.fetch ?? fetch;
    this.#signer = options.signer;
    this.#timeoutMilliseconds = timeoutMilliseconds;
  }

  async loadInvite(code: string): Promise<ResolvedInvite> {
    const inviteCode = validInviteCode(code);
    await this.#negotiate("read");

    const resolved = await this.#postJson(INVITE_RESOLVE_PATH, {
      code: inviteCode,
    });
    const status = inviteStatus(resolved.status);
    if (status !== "active") throw inviteStatusError(status);

    const communityId = requiredString(resolved, "community_id");
    const policy = await this.#postJson(communityJoinPolicyPath(communityId), {
      invite_code: inviteCode,
    });

    return {
      communityId,
      communityHost: requiredString(resolved, "community_host"),
      status,
      role: inviteRole(resolved.role),
      joinPolicy: parseJoinPolicy(policy.policy),
    };
  }

  async redeemInvite(
    code: string,
    policyAcceptance?: InvitePolicyAcceptance,
  ): Promise<InviteRedemption> {
    const inviteCode = validInviteCode(code);
    await this.#negotiate("write");

    const url = this.#url(INVITE_REDEEM_PATH);
    const body = JSON.stringify({
      code: inviteCode,
      policy_acceptance: policyAcceptance
        ? {
            policy_version: policyAcceptance.policyVersion,
            age_confirmed: policyAcceptance.ageConfirmed,
            legal_documents_accepted: policyAcceptance.legalDocumentsAccepted,
          }
        : undefined,
    });
    const authorization = await makeNip98Authorization(
      this.#signer,
      url.toString(),
      "POST",
      body,
    );
    const response = await this.#request(url, {
      method: "POST",
      headers: {
        Authorization: authorization,
        "Content-Type": "application/json",
      },
      body,
      redirect: "error",
    });
    const json = await readJsonObject(response);
    if (!response.ok) throw responseError(response.status, json);

    const status = json.status;
    if (status !== "joined" && status !== "already_member") {
      throw invalidResponse();
    }
    return {
      status,
      communityId: requiredString(json, "community_id"),
      communityHost: requiredString(json, "community_host"),
      role: inviteRole(json.role),
    };
  }

  async #negotiate(access: CompatibilityAccess): Promise<void> {
    const response = await this.#postJson(COMPATIBILITY_PATH, {
      client_id: WEB_CLIENT_ID,
      client_version: WEB_CLIENT_VERSION,
      access,
      protocols: [{ id: "collaboration-http", version: 1 }],
      features: [...REQUIRED_FEATURES],
    });
    const compatibility = response as unknown as CompatibilityResponse;
    if (
      compatibility.outcome === "upgrade_required" ||
      compatibility.error === "upgrade_required"
    ) {
      const minimum = optionalString(compatibility.minimum_client_version);
      const maximum = optionalString(compatibility.maximum_client_version);
      throw new CollaborationAuthError(
        "upgrade_required",
        versionMessage(minimum, maximum),
        { minimumVersion: minimum, maximumVersion: maximum },
      );
    }
    if (access === "write" && compatibility.outcome === "read_only") {
      throw new CollaborationAuthError(
        "read_only",
        "This client can view the invite but cannot accept it. Upgrade before continuing.",
      );
    }
    if (
      !Number.isInteger(compatibility.policy_version) ||
      compatibility.policy_version < MINIMUM_COMPATIBILITY_POLICY_VERSION ||
      compatibility.outcome !== "supported" ||
      compatibility.client_id !== WEB_CLIENT_ID ||
      compatibility.retryable !== false ||
      !Array.isArray(compatibility.selected_features) ||
      REQUIRED_FEATURES.some(
        (feature) => !compatibility.selected_features.includes(feature),
      )
    ) {
      throw invalidResponse();
    }
  }

  async #postJson(path: string, body: JsonObject): Promise<JsonObject> {
    const response = await this.#request(this.#url(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      redirect: "error",
    });
    const json = await readJsonObject(response);
    if (!response.ok) throw responseError(response.status, json);
    return json;
  }

  async #request(url: URL, init: RequestInit): Promise<Response> {
    try {
      return await this.#fetch(url, {
        ...init,
        signal: AbortSignal.timeout(this.#timeoutMilliseconds),
      });
    } catch (error) {
      if (error instanceof CollaborationAuthError) throw error;
      throw new CollaborationAuthError(
        "service_unavailable",
        "The collaboration service is unavailable. Try again.",
        { retryable: true },
      );
    }
  }

  #url(path: string): URL {
    return new URL(path, this.#baseUrl);
  }
}

async function readJsonObject(response: Response): Promise<JsonObject> {
  const text = await response.text();
  if (text.length > MAX_RESPONSE_CHARACTERS) throw invalidResponse();
  try {
    const value: unknown = JSON.parse(text);
    if (value === null || Array.isArray(value) || typeof value !== "object") {
      throw invalidResponse();
    }
    return value as JsonObject;
  } catch (error) {
    if (error instanceof CollaborationAuthError) throw error;
    throw invalidResponse();
  }
}

function responseError(
  status: number,
  response: JsonObject,
): CollaborationAuthError {
  const error = optionalString(response.error);
  if (error === "upgrade_required" || status === 426) {
    const minimum = optionalString(response.minimum_client_version);
    const maximum = optionalString(response.maximum_client_version);
    return new CollaborationAuthError(
      "upgrade_required",
      versionMessage(minimum, maximum),
      { minimumVersion: minimum, maximumVersion: maximum },
    );
  }
  if (error === "invite_expired") return inviteStatusError("expired");
  if (error === "invite_exhausted") return inviteStatusError("exhausted");
  if (error === "invite_revoked") return inviteStatusError("revoked");
  if (error === "invite_invalid") return inviteStatusError("invalid");
  return new CollaborationAuthError(
    "service_unavailable",
    "The collaboration service could not complete the invite request.",
    { retryable: status >= 500 },
  );
}

function inviteStatus(value: unknown): InviteStatus {
  if (
    value === "active" ||
    value === "expired" ||
    value === "exhausted" ||
    value === "revoked"
  ) {
    return value;
  }
  throw invalidResponse();
}

function inviteStatusError(
  status: Exclude<InviteStatus, "active"> | "invalid",
): CollaborationAuthError {
  const details = {
    expired: [
      "invite_expired",
      "This invite has expired. Ask for a new invite.",
    ],
    exhausted: [
      "invite_exhausted",
      "This invite has reached its use limit. Ask for a new invite.",
    ],
    revoked: [
      "invite_revoked",
      "This invite was revoked. Ask for a new invite.",
    ],
    invalid: [
      "invite_invalid",
      "This invite is invalid. Check the link or ask for a new invite.",
    ],
  } as const;
  const [kind, message] = details[status];
  return new CollaborationAuthError(kind, message);
}

function parseJoinPolicy(value: unknown): JoinPolicy | null {
  if (value === null) return null;
  if (
    value === undefined ||
    Array.isArray(value) ||
    typeof value !== "object"
  ) {
    throw invalidResponse();
  }
  const policy = value as JsonObject;
  if (typeof policy.age_attestation_required !== "boolean") {
    throw invalidResponse();
  }
  return {
    version: requiredString(policy, "version"),
    age_attestation_required: policy.age_attestation_required,
    terms_markdown: optionalString(policy.terms_markdown),
    privacy_markdown: optionalString(policy.privacy_markdown),
  };
}

function inviteRole(value: unknown): "member" | "guest" {
  if (value === "member" || value === "guest") return value;
  throw invalidResponse();
}

function requiredString(object: JsonObject, key: string): string {
  const value = object[key];
  if (typeof value !== "string" || value.length === 0 || value.length > 2_048) {
    throw invalidResponse();
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 && value.length <= 2_048
    ? value
    : undefined;
}

function versionMessage(minimum?: string, maximum?: string): string {
  if (minimum && maximum) {
    return `Buzz web ${WEB_CLIENT_VERSION} is unsupported. Use version ${minimum} through ${maximum}.`;
  }
  return "This Buzz web version is unsupported. Upgrade before continuing.";
}

function invalidResponse(): CollaborationAuthError {
  return new CollaborationAuthError(
    "invalid_response",
    "The collaboration service returned an invalid response.",
  );
}

function invalidServiceUrl(): CollaborationAuthError {
  return new CollaborationAuthError(
    "invalid_response",
    "The collaboration service URL is invalid.",
  );
}
