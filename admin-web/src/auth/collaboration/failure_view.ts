import { AdminCollaborationError } from "../../data/collaboration/contracts.ts";
import { AdminSessionError } from "./session_transport.ts";

export type AdminFailureAction = "retry" | "reload" | "sign_in" | "upgrade";

export type AdminFailureView = {
  title: string;
  message: string;
  action?: AdminFailureAction;
  actionLabel?: string;
  preserveTrustedData: boolean;
  role: "alert" | "status";
};

export type AdminResourceState<T> =
  | { status: "idle"; data?: undefined; failure?: undefined }
  | { status: "loading"; data?: T; failure?: undefined }
  | { status: "ready"; data: T; failure?: undefined }
  | { status: "partial"; data: T; failure: AdminFailureView }
  | { status: "error"; data?: undefined; failure: AdminFailureView };

export function adminFailureView(error: unknown, hasTrustedData: boolean): AdminFailureView {
  if (error instanceof AdminSessionError) {
    if (error.reason === "expired") {
      return {
        title: "Session expired",
        message: "Sign in again to continue administration.",
        action: "sign_in",
        actionLabel: "Sign in",
        preserveTrustedData: hasTrustedData,
        role: "alert",
      };
    }
    if (error.reason === "unavailable") {
      return unavailableView(hasTrustedData);
    }
    return deniedView(hasTrustedData);
  }
  if (!(error instanceof AdminCollaborationError)) {
    return unavailableView(hasTrustedData);
  }
  switch (error.kind) {
    case "authorization_denied":
      return deniedView(hasTrustedData);
    case "upgrade_required":
      return {
        title: "Upgrade required",
        message: "Upgrade this administration client before continuing.",
        action: "upgrade",
        actionLabel: "Review update",
        preserveTrustedData: hasTrustedData,
        role: "alert",
      };
    case "stale_write":
      return {
        title: "Data changed",
        message: "Reload the resource before trying this action again.",
        action: "reload",
        actionLabel: "Reload",
        preserveTrustedData: hasTrustedData,
        role: "status",
      };
    case "outcome_unknown":
      return {
        title: "Action status unknown",
        message: "Reload the last trustworthy state before taking another action.",
        action: "reload",
        actionLabel: "Reload",
        preserveTrustedData: hasTrustedData,
        role: "alert",
      };
    case "service_unavailable":
      return unavailableView(hasTrustedData);
    case "resource_unavailable":
    case "invalid_request":
    case "invalid_response":
      return {
        title: "Could not load administration data",
        message: "The requested administration data is unavailable.",
        action: "retry",
        actionLabel: "Retry",
        preserveTrustedData: hasTrustedData,
        role: "alert",
      };
  }
}

export class AdminResourceController<T> {
  readonly #loadResource: () => Promise<T>;
  #state: AdminResourceState<T> = { status: "idle" };

  constructor(loadResource: () => Promise<T>) {
    this.#loadResource = loadResource;
  }

  get state(): AdminResourceState<T> {
    return this.#state;
  }

  async load(): Promise<AdminResourceState<T>> {
    const trustedData = this.#state.data;
    this.#state = { status: "loading", data: trustedData };
    try {
      const data = await this.#loadResource();
      this.#state = { status: "ready", data };
    } catch (error) {
      const failure = adminFailureView(error, trustedData !== undefined);
      this.#state =
        trustedData === undefined ? { status: "error", failure } : { status: "partial", data: trustedData, failure };
    }
    return this.#state;
  }

  async retry(): Promise<AdminResourceState<T>> {
    return this.load();
  }
}

function deniedView(hasTrustedData: boolean): AdminFailureView {
  return {
    title: "Access denied",
    message: "Your operator role does not permit this administration action.",
    preserveTrustedData: hasTrustedData,
    role: "alert",
  };
}

function unavailableView(hasTrustedData: boolean): AdminFailureView {
  return {
    title: hasTrustedData ? "Some administration data is unavailable" : "Could not load administration data",
    message: hasTrustedData
      ? "Showing the last trustworthy data. Retry the unavailable operation."
      : "The administration service is temporarily unavailable.",
    action: "retry",
    actionLabel: "Retry",
    preserveTrustedData: hasTrustedData,
    role: "alert",
  };
}
