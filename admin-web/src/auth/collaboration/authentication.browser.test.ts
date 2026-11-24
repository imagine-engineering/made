import assert from "node:assert/strict";
import test from "node:test";

import { AdminCollaborationClient } from "../../data/collaboration/client.ts";
import {
  ADMIN_CLIENT_ID,
  COMPATIBILITY_PATH,
  AdminCollaborationError,
  type AdminCollaborationTransport,
  type AdminTransportRequest,
  type AdminTransportResponse,
} from "../../data/collaboration/contracts.ts";
import { AdminResourceController, adminFailureView } from "./failure_view.ts";
import {
  AdminSessionError,
  AdminSessionTransport,
  type AdminCredentialedTransport,
  type AdminOperatorSession,
  type AdminOperatorSessionProvider,
} from "./session_transport.ts";

const NOW_MILLIS = 1_787_600_000_000;
const COMMUNITY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057001";
const PRINCIPAL_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057002";
const TENANT_CANARY = "private-engineering.example.test";

class StaticSessionProvider implements AdminOperatorSessionProvider {
  readonly session: AdminOperatorSession | null;

  constructor(session: AdminOperatorSession | null) {
    this.session = session;
  }

  async currentSession(): Promise<AdminOperatorSession | null> {
    return this.session;
  }
}

class CompatibilityTransport implements AdminCollaborationTransport {
  readonly requests: AdminTransportRequest[] = [];

  async request(request: AdminTransportRequest): Promise<AdminTransportResponse> {
    this.requests.push(structuredClone(request));
    assert.equal(request.path, COMPATIBILITY_PATH);
    return {
      status: 200,
      body: {
        policy_version: 1,
        outcome: "supported",
        client_id: ADMIN_CLIENT_ID,
        selected_features: request.body?.features,
        retryable: false,
      },
    };
  }
}

class CredentialedService implements AdminCredentialedTransport {
  readonly requests: AdminTransportRequest[] = [];
  calls = 0;
  mode: "supported" | "denied" | "unavailable" = "supported";
  measuredAtMillis = NOW_MILLIS;

  async request(credentialReference: string, request: AdminTransportRequest): Promise<AdminTransportResponse> {
    this.calls += 1;
    this.requests.push(structuredClone(request));
    assert.equal(credentialReference, "credential-reference-7");
    if (this.mode === "denied") {
      return {
        status: 403,
        body: {
          error: "authorization_denied",
          tenant: TENANT_CANARY,
          message: `operator cannot access ${TENANT_CANARY}`,
        },
      };
    }
    if (this.mode === "unavailable") {
      return { status: 503, body: { error: "service_unavailable" } };
    }
    if (request.path.endsWith("/moderation/reports")) {
      return { status: 200, body: { reports: [] } };
    }
    if (request.path.endsWith("/metrics")) {
      return {
        status: 200,
        body: {
          active_members: 42,
          open_reports: 2,
          pending_invites: 3,
          deletion_completed_phases: 0,
          deletion_total_phases: 6,
          measured_at_millis: this.measuredAtMillis,
        },
      };
    }
    return { status: 404, body: { error: "not_found" } };
  }
}

function session(overrides: Partial<AdminOperatorSession> = {}): AdminOperatorSession {
  return {
    communityId: COMMUNITY_ID,
    principalId: PRINCIPAL_ID,
    role: "admin",
    scopes: ["communities:manage", "moderation:manage"],
    expiresAtMillis: NOW_MILLIS + 60_000,
    credentialReference: "credential-reference-7",
    ...overrides,
  };
}

function clientFor(
  operatorSession: AdminOperatorSession | null,
  credentialedService = new CredentialedService(),
): {
  client: AdminCollaborationClient;
  credentialedService: CredentialedService;
  compatibility: CompatibilityTransport;
} {
  const compatibility = new CompatibilityTransport();
  const transport = new AdminSessionTransport({
    publicTransport: compatibility,
    credentialedTransport: credentialedService,
    sessionProvider: new StaticSessionProvider(operatorSession),
    nowMillis: () => NOW_MILLIS,
  });
  return {
    client: new AdminCollaborationClient(transport),
    credentialedService,
    compatibility,
  };
}

test("denied roles fail before the credentialed administration request", async () => {
  const { client, credentialedService, compatibility } = clientFor(
    session({ role: "member", scopes: ["moderation:manage"] }),
  );

  await assert.rejects(client.listModerationReports(COMMUNITY_ID), (error) => {
    assert.ok(error instanceof AdminSessionError);
    assert.equal(error.reason, "denied");
    assert.equal(error.kind, "authorization_denied");
    return true;
  });
  assert.equal(compatibility.requests.length, 1);
  assert.equal(credentialedService.calls, 0);
  assert.deepEqual(adminFailureView(new AdminSessionError("denied"), false), {
    title: "Access denied",
    message: "Your operator role does not permit this administration action.",
    preserveTrustedData: false,
    role: "alert",
  });
});

test("expired sessions request sign-in without sending a credential", async () => {
  const { client, credentialedService } = clientFor(session({ expiresAtMillis: NOW_MILLIS }));

  let failure: AdminSessionError | undefined;
  await assert.rejects(client.getMetrics(COMMUNITY_ID), (error) => {
    assert.ok(error instanceof AdminSessionError);
    failure = error;
    return error.reason === "expired";
  });
  assert.equal(credentialedService.calls, 0);
  assert.equal(adminFailureView(failure, false).action, "sign_in");
});

test("partial service failures retain trustworthy data and retry cleanly", async () => {
  const credentialedService = new CredentialedService();
  const { client } = clientFor(session(), credentialedService);
  const controller = new AdminResourceController(() => client.getMetrics(COMMUNITY_ID));

  const ready = await controller.load();
  assert.equal(ready.status, "ready");
  assert.equal(ready.data?.activeMembers, 42);

  credentialedService.mode = "unavailable";
  const partial = await controller.retry();
  assert.equal(partial.status, "partial");
  assert.equal(partial.data.activeMembers, 42);
  assert.equal(partial.failure.action, "retry");
  assert.equal(partial.failure.preserveTrustedData, true);

  credentialedService.mode = "supported";
  credentialedService.measuredAtMillis += 1;
  const recovered = await controller.retry();
  assert.equal(recovered.status, "ready");
  assert.equal(recovered.data.measuredAtMillis, NOW_MILLIS + 1);
});

test("server diagnostics cannot disclose tenant metadata through failure UX", async () => {
  const credentialedService = new CredentialedService();
  credentialedService.mode = "denied";
  const { client } = clientFor(session(), credentialedService);
  const controller = new AdminResourceController(() => client.getMetrics(COMMUNITY_ID));

  const state = await controller.load();
  assert.equal(state.status, "error");
  assert.equal(state.failure.title, "Access denied");
  assert.equal(JSON.stringify(state).includes(TENANT_CANARY), false);
  assert.equal(JSON.stringify(state).includes(COMMUNITY_ID), false);

  const rawError = new AdminCollaborationError("authorization_denied", `denied for ${TENANT_CANARY}`);
  assert.equal(JSON.stringify(adminFailureView(rawError, false)).includes(TENANT_CANARY), false);
});
