import {
  COMPATIBILITY_PATH,
  AdminCollaborationError,
  validUuid,
  type AdminCollaborationErrorKind,
  type AdminCollaborationTransport,
  type AdminScope,
  type AdminTransportRequest,
  type AdminTransportResponse,
} from "../../data/collaboration/contracts.ts";

const MAX_CREDENTIAL_REFERENCE_CHARACTERS = 256;

export type AdminOperatorRole = "owner" | "admin" | "member" | "guest" | "bot";

export type AdminOperatorSession = {
  communityId: string;
  principalId: string;
  role: AdminOperatorRole;
  scopes: readonly AdminScope[];
  expiresAtMillis: number;
  credentialReference: string;
};

export interface AdminOperatorSessionProvider {
  currentSession(): Promise<AdminOperatorSession | null>;
}

export interface AdminCredentialedTransport {
  request(credentialReference: string, request: AdminTransportRequest): Promise<AdminTransportResponse>;
}

export type AdminSessionFailureReason = "denied" | "expired" | "unavailable";

export class AdminSessionError extends AdminCollaborationError {
  readonly reason: AdminSessionFailureReason;

  constructor(reason: AdminSessionFailureReason) {
    const unavailable = reason === "unavailable";
    const kind: AdminCollaborationErrorKind = unavailable ? "service_unavailable" : "authorization_denied";
    super(
      kind,
      unavailable ? "The operator session service is unavailable." : "The administration request was denied.",
      { retryable: unavailable },
    );
    this.name = "AdminSessionError";
    this.reason = reason;
  }
}

export class AdminSessionTransport implements AdminCollaborationTransport {
  readonly #publicTransport: AdminCollaborationTransport;
  readonly #credentialedTransport: AdminCredentialedTransport;
  readonly #sessionProvider: AdminOperatorSessionProvider;
  readonly #nowMillis: () => number;

  constructor(options: {
    publicTransport: AdminCollaborationTransport;
    credentialedTransport: AdminCredentialedTransport;
    sessionProvider: AdminOperatorSessionProvider;
    nowMillis?: () => number;
  }) {
    this.#publicTransport = options.publicTransport;
    this.#credentialedTransport = options.credentialedTransport;
    this.#sessionProvider = options.sessionProvider;
    this.#nowMillis = options.nowMillis ?? Date.now;
  }

  async request(request: AdminTransportRequest): Promise<AdminTransportResponse> {
    if (request.path === COMPATIBILITY_PATH) {
      if (request.requiredScope !== undefined) throw invalidRequest();
      return this.#publicTransport.request(request);
    }
    if (request.requiredScope === undefined) throw invalidRequest();

    let session: AdminOperatorSession | null;
    try {
      session = await this.#sessionProvider.currentSession();
    } catch {
      throw new AdminSessionError("unavailable");
    }
    if (session === null) throw new AdminSessionError("expired");
    validateSession(session);
    if (session.expiresAtMillis <= this.#nowMillis()) {
      throw new AdminSessionError("expired");
    }
    const communityId = requestCommunityId(request.path);
    if (
      communityId !== session.communityId ||
      !session.scopes.includes(request.requiredScope) ||
      (session.role !== "owner" && session.role !== "admin")
    ) {
      throw new AdminSessionError("denied");
    }
    try {
      return await this.#credentialedTransport.request(session.credentialReference, request);
    } catch (error) {
      if (error instanceof AdminCollaborationError) throw error;
      throw new AdminSessionError("unavailable");
    }
  }
}

function validateSession(session: AdminOperatorSession): void {
  try {
    validUuid(session.communityId);
    validUuid(session.principalId);
  } catch {
    throw new AdminSessionError("denied");
  }
  if (
    !Number.isSafeInteger(session.expiresAtMillis) ||
    session.scopes.length === 0 ||
    session.scopes.length > 2 ||
    new Set(session.scopes).size !== session.scopes.length ||
    !session.scopes.every((scope) => scope === "communities:manage" || scope === "moderation:manage") ||
    session.credentialReference.length === 0 ||
    session.credentialReference.length > MAX_CREDENTIAL_REFERENCE_CHARACTERS ||
    containsControlCharacter(session.credentialReference)
  ) {
    throw new AdminSessionError("denied");
  }
}

function requestCommunityId(path: string): string {
  const match = /^\/v1\/collaboration\/admin\/communities\/([^/]+)(?:\/|$)/.exec(path);
  if (match?.[1] === undefined) throw invalidRequest();
  try {
    return validUuid(decodeURIComponent(match[1]));
  } catch {
    throw invalidRequest();
  }
}

function containsControlCharacter(value: string): boolean {
  return Array.from(value).some((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint !== undefined && (codePoint <= 31 || codePoint === 127);
  });
}

function invalidRequest(): AdminCollaborationError {
  return new AdminCollaborationError("invalid_request", "The administration request is invalid.");
}
