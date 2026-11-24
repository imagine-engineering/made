import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { adminFailureView } from "../../src/auth/collaboration/failure_view.ts";
import {
  AdminSessionTransport,
  type AdminCredentialedTransport,
  type AdminOperatorSession,
  type AdminOperatorSessionProvider,
} from "../../src/auth/collaboration/session_transport.ts";
import { AdminCollaborationClient } from "../../src/data/collaboration/client.ts";
import {
  ADMIN_CLIENT_ID,
  ADMIN_CLIENT_VERSION,
  COMPATIBILITY_PATH,
  AdminCollaborationError,
  type AdminCollaborationTransport,
  type AdminTransportRequest,
  type AdminTransportResponse,
  type JsonObject,
} from "../../src/data/collaboration/contracts.ts";

const NOW_MILLIS = 1_787_600_000_000;
const COMMUNITY_ID = operationId(1);
const FOREIGN_COMMUNITY_ID = operationId(2);
const OWNER_ID = operationId(3);
const MEMBER_ID = operationId(4);
const INVITE_ID = operationId(5);
const REPORT_ID = operationId(6);
const DELETION_ID = operationId(7);
const SUPPORTED_SERVICE_VERSIONS = ["0.44.0"] as const;
const TENANT_CANARY = "secret-community.example.test";
const manifest = JSON.parse(
  readFileSync(
    new URL("../../../.agents/specs/collaborative-workspace/fixtures/clients/manifest.json", import.meta.url),
    "utf8",
  ),
) as ClientManifest;

type ClientManifest = {
  clients: Record<string, { version: string }>;
  contracts: Array<{
    id: string;
    expected_output: JsonObject;
  }>;
};

type ServiceMode = "supported" | "upgrade_required";

class CanonicalAdminService
  implements AdminCollaborationTransport, AdminCredentialedTransport, AdminOperatorSessionProvider
{
  readonly serviceVersion: string;
  readonly compatibilityRequests: AdminTransportRequest[] = [];
  readonly resourceRequests: AdminTransportRequest[] = [];
  mode: ServiceMode = "supported";
  session: AdminOperatorSession;
  community = {
    displayName: "Engineering",
    state: "active" as "active" | "archived",
    version: 1,
  };
  member = { role: "member", state: "active", version: 1 };
  invite = { state: "absent" as "absent" | "active" | "revoked", version: 0 };
  report = { state: "open" as "open" | "resolved", version: 1 };
  deletion = {
    stage: "reversible" as "reversible" | "rolled_back",
    version: 2,
  };

  constructor(serviceVersion: string, role: AdminOperatorSession["role"] = "owner") {
    this.serviceVersion = serviceVersion;
    this.session = {
      communityId: COMMUNITY_ID,
      principalId: OWNER_ID,
      role,
      scopes: ["communities:manage", "moderation:manage"],
      expiresAtMillis: NOW_MILLIS + 60_000,
      credentialReference: "operator-session-reference",
    };
  }

  async currentSession(): Promise<AdminOperatorSession> {
    return this.session;
  }

  async request(
    requestOrCredential: AdminTransportRequest | string,
    credentialedRequest?: AdminTransportRequest,
  ): Promise<AdminTransportResponse> {
    if (typeof requestOrCredential === "string") {
      assert.equal(requestOrCredential, "operator-session-reference");
      assert.ok(credentialedRequest);
      this.resourceRequests.push(structuredClone(credentialedRequest));
      return this.resourceResponse(credentialedRequest);
    }
    const request = requestOrCredential;
    assert.equal(request.path, COMPATIBILITY_PATH);
    assert.equal(request.requiredScope, undefined);
    this.compatibilityRequests.push(structuredClone(request));
    const body = request.body ?? {};
    const supported =
      this.mode === "supported" &&
      body.client_id === ADMIN_CLIENT_ID &&
      body.client_version === ADMIN_CLIENT_VERSION &&
      this.serviceVersion === "0.44.0";
    if (!supported) {
      return {
        status: 426,
        body: {
          policy_version: 1,
          outcome: "upgrade_required",
          error: "upgrade_required",
          client_id: ADMIN_CLIENT_ID,
          minimum_client_version: ADMIN_CLIENT_VERSION,
          maximum_client_version: ADMIN_CLIENT_VERSION,
          minimum_service_version: "0.44.0",
          maximum_service_version: "0.44.0",
          selected_features: [],
          retryable: false,
        },
      };
    }
    return {
      status: 200,
      body: {
        policy_version: 1,
        outcome: "supported",
        client_id: ADMIN_CLIENT_ID,
        service_version: this.serviceVersion,
        selected_features: body.features,
        retryable: false,
      },
    };
  }

  private resourceResponse(request: AdminTransportRequest): AdminTransportResponse {
    const base = `/v1/collaboration/admin/communities/${COMMUNITY_ID}`;
    if (request.path === base && request.method === "GET") {
      return ok({
        community_id: COMMUNITY_ID,
        display_name: this.community.displayName,
        state: this.community.state,
        version: this.community.version,
      });
    }
    if (request.path === base && request.method === "PATCH") {
      if (!this.matchesVersion(request, this.community.version)) return stale();
      this.community.displayName = request.body?.display_name as string;
      this.community.version += 1;
      return this.receipt(request, COMMUNITY_ID, this.community.version);
    }
    if (request.path === `${base}/members` && request.method === "GET") {
      return ok({
        members: [
          {
            principal_id: MEMBER_ID,
            role: this.member.role,
            state: this.member.state,
            version: this.member.version,
          },
        ],
      });
    }
    if (request.path === `${base}/members/${MEMBER_ID}` && request.method === "PATCH") {
      if (!this.matchesVersion(request, this.member.version)) return stale();
      this.member.role = request.body?.role as string;
      this.member.version += 1;
      return this.receipt(request, MEMBER_ID, this.member.version);
    }
    if (request.path === `${base}/invites` && request.method === "GET") {
      return ok({
        invites:
          this.invite.state === "absent"
            ? []
            : [
                {
                  invite_id: INVITE_ID,
                  role: "member",
                  state: this.invite.state,
                  remaining_uses: this.invite.state === "active" ? 3 : 0,
                  version: this.invite.version,
                },
              ],
      });
    }
    if (request.path === `${base}/invites` && request.method === "POST") {
      this.invite = { state: "active", version: 1 };
      return this.receipt(request, INVITE_ID, 1);
    }
    if (request.path === `${base}/invites/${INVITE_ID}/revoke` && request.method === "POST") {
      if (!this.matchesVersion(request, this.invite.version)) return stale();
      this.invite = { state: "revoked", version: 2 };
      return this.receipt(request, INVITE_ID, 2);
    }
    if (request.path === `${base}/moderation/reports` && request.method === "GET") {
      return ok({
        reports: [
          {
            report_id: REPORT_ID,
            target_kind: "event",
            target_id: "ab".repeat(32),
            reason: "spam",
            state: this.report.state,
            version: this.report.version,
          },
        ],
      });
    }
    if (request.path === `${base}/moderation/reports/${REPORT_ID}/resolve` && request.method === "POST") {
      if (!this.matchesVersion(request, this.report.version)) return stale();
      this.report = { state: "resolved", version: 2 };
      return this.receipt(request, REPORT_ID, 2);
    }
    if (request.path === `${base}/archive` && request.method === "POST") {
      if (this.session.role !== "owner") return denied();
      if (!this.matchesVersion(request, this.community.version)) return stale();
      this.community.state = "archived";
      this.community.version += 1;
      return this.receipt(request, COMMUNITY_ID, this.community.version);
    }
    if (request.path === `${base}/deletions/${DELETION_ID}` && request.method === "GET") {
      return ok(this.deletionStatus());
    }
    if (request.path === `${base}/deletions/${DELETION_ID}/restore` && request.method === "POST") {
      if (this.session.role !== "owner") return denied();
      if (!this.matchesVersion(request, this.deletion.version)) return stale();
      this.deletion = { stage: "rolled_back", version: 3 };
      return ok(this.deletionStatus());
    }
    if (request.path === `${base}/metrics` && request.method === "GET") {
      return ok({
        active_members: 2,
        open_reports: this.report.state === "open" ? 1 : 0,
        pending_invites: this.invite.state === "active" ? 1 : 0,
        deletion_completed_phases: 0,
        deletion_total_phases: 6,
        measured_at_millis: NOW_MILLIS,
      });
    }
    return { status: 404, body: { error: "not_found" } };
  }

  private matchesVersion(request: AdminTransportRequest, version: number): boolean {
    return request.body?.expected_version === version;
  }

  private receipt(request: AdminTransportRequest, resourceId: string, version: number): AdminTransportResponse {
    return ok({
      operation_id: request.body?.operation_id,
      resource_id: resourceId,
      version,
    });
  }

  private deletionStatus(): JsonObject {
    return this.deletion.stage === "reversible"
      ? {
          stage: "reversible",
          last_trustworthy_stage: "reversible",
          completed_phases: 0,
          total_phases: 6,
          next_phase: "database",
          recovery_action: "restore",
          authority_evidence: TENANT_CANARY,
        }
      : {
          stage: "rolled_back",
          last_trustworthy_stage: "rolled_back",
          completed_phases: 0,
          total_phases: 6,
          recovery_action: "none",
          authority_evidence: TENANT_CANARY,
        };
  }
}

function clientFor(service: CanonicalAdminService): AdminCollaborationClient {
  return new AdminCollaborationClient(
    new AdminSessionTransport({
      publicTransport: service,
      credentialedTransport: service,
      sessionProvider: service,
      nowMillis: () => NOW_MILLIS,
    }),
  );
}

function contract(id: string): ClientManifest["contracts"][number] {
  const value = manifest.contracts.find((candidate) => candidate.id === id);
  assert.ok(value, `missing frozen contract ${id}`);
  return value;
}

for (const serviceVersion of SUPPORTED_SERVICE_VERSIONS) {
  test(`Buzz admin ${ADMIN_CLIENT_VERSION} completes canonical lifecycle against Collab ${serviceVersion}`, async () => {
    assert.equal(manifest.clients[ADMIN_CLIENT_ID]?.version, ADMIN_CLIENT_VERSION);
    assert.deepEqual(contract("CLIENT-ADMIN-001").expected_output.routes, [
      "/reports",
      "/reports/:id",
      "/feedback",
      "/feedback/:id",
    ]);
    const service = new CanonicalAdminService(serviceVersion);
    const client = clientFor(service);

    assert.equal((await client.getCommunity(COMMUNITY_ID)).version, 1);
    assert.equal((await client.updateCommunity(COMMUNITY_ID, 1, operationId(10), "Platform Engineering")).version, 2);
    assert.equal((await client.listMembers(COMMUNITY_ID))[0]?.role, "member");
    assert.equal((await client.updateMemberRole(COMMUNITY_ID, MEMBER_ID, "admin", 1, operationId(11))).version, 2);
    assert.deepEqual(await client.listInvites(COMMUNITY_ID), []);
    assert.equal(
      (await client.createInvite(COMMUNITY_ID, "member", NOW_MILLIS + 60_000, 3, operationId(12))).resourceId,
      INVITE_ID,
    );
    assert.equal((await client.listInvites(COMMUNITY_ID))[0]?.state, "active");
    assert.equal((await client.revokeInvite(COMMUNITY_ID, INVITE_ID, 1, operationId(13))).version, 2);
    assert.equal((await client.listModerationReports(COMMUNITY_ID))[0]?.state, "open");
    assert.equal(
      (await client.resolveModerationReport(COMMUNITY_ID, REPORT_ID, 1, "actioned", operationId(14))).version,
      2,
    );
    assert.equal((await client.archiveCommunity(COMMUNITY_ID, 2, operationId(15))).version, 3);
    const metrics = await client.getMetrics(COMMUNITY_ID);
    assert.equal(metrics.openReports, 0);
    assert.equal(metrics.pendingInvites, 0);

    assert.equal(service.compatibilityRequests.length, service.resourceRequests.length);
    assert.ok(
      service.compatibilityRequests.every(
        (request) =>
          request.body?.client_id === ADMIN_CLIENT_ID && request.body?.client_version === ADMIN_CLIENT_VERSION,
      ),
    );
  });
}

test("owner-only and cross-tenant operations fail without disclosure", async () => {
  const service = new CanonicalAdminService("0.44.0", "admin");
  const client = clientFor(service);

  await assert.rejects(
    client.archiveCommunity(COMMUNITY_ID, 1, operationId(20)),
    (error) => error instanceof AdminCollaborationError && error.kind === "authorization_denied",
  );
  const callsAfterRoleDenial = service.resourceRequests.length;
  await assert.rejects(
    client.getMetrics(FOREIGN_COMMUNITY_ID),
    (error) => error instanceof AdminCollaborationError && error.kind === "authorization_denied",
  );
  assert.equal(service.resourceRequests.length, callsAfterRoleDenial);
  assert.equal(
    JSON.stringify(
      adminFailureView(new AdminCollaborationError("authorization_denied", `denied for ${TENANT_CANARY}`), false),
    ).includes(TENANT_CANARY),
    false,
  );
});

test("reversible deletion recovery returns only the operator-safe projection", async () => {
  const service = new CanonicalAdminService("0.44.0");
  const client = clientFor(service);

  const before = await client.getDeletionStatus(COMMUNITY_ID, DELETION_ID);
  assert.equal(before.stage, "reversible");
  assert.equal(before.recoveryAction, "restore");
  assert.equal(JSON.stringify(before).includes(TENANT_CANARY), false);
  const restored = await client.restoreDeletion(COMMUNITY_ID, DELETION_ID, 2, operationId(21));
  assert.equal(restored.stage, "rolled_back");
  assert.equal(restored.recoveryAction, "none");
  assert.equal(JSON.stringify(restored).includes(TENANT_CANARY), false);
});

test("unsupported versions stop before session and resource access", async () => {
  const service = new CanonicalAdminService("0.45.0");
  let sessionLookups = 0;
  const sessionProvider: AdminOperatorSessionProvider = {
    async currentSession() {
      sessionLookups += 1;
      return service.session;
    },
  };
  const client = new AdminCollaborationClient(
    new AdminSessionTransport({
      publicTransport: service,
      credentialedTransport: service,
      sessionProvider,
      nowMillis: () => NOW_MILLIS,
    }),
  );

  await assert.rejects(client.getCommunity(COMMUNITY_ID), (error) => {
    assert.ok(error instanceof AdminCollaborationError);
    assert.equal(error.kind, "upgrade_required");
    assert.equal(error.minimumVersion, ADMIN_CLIENT_VERSION);
    return true;
  });
  assert.equal(sessionLookups, 0);
  assert.equal(service.resourceRequests.length, 0);
});

test("frozen forbidden and unavailable-content UX remains represented", () => {
  const deniedContract = contract("CLIENT-ADMIN-002").expected_output;
  assert.equal(deniedContract.heading, "Access denied");
  assert.equal(deniedContract.retry_action, true);
  assert.equal(deniedContract.credentials, "same-origin");
  assert.equal(
    contract("CLIENT-ADMIN-003").expected_output.message,
    "Message content is unavailable. It may have expired or been removed from event storage.",
  );
  assert.equal(contract("CLIENT-ADMIN-004").expected_output.scope, "local-browser-only");
});

function operationId(value: number): string {
  return `018fbe5f-6f37-7b40-8fb3-${value.toString().padStart(12, "0")}`;
}

function ok(body: JsonObject): AdminTransportResponse {
  return { status: 200, body };
}

function stale(): AdminTransportResponse {
  return { status: 409, body: { error: "stale_write" } };
}

function denied(): AdminTransportResponse {
  return {
    status: 403,
    body: {
      error: "authorization_denied",
      tenant: TENANT_CANARY,
      message: `not permitted in ${TENANT_CANARY}`,
    },
  };
}
