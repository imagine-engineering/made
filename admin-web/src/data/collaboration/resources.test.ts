import assert from "node:assert/strict";
import test from "node:test";

import { AdminCollaborationClient } from "./client.ts";
import {
  ADMIN_CLIENT_ID,
  COMPATIBILITY_PATH,
  AdminCollaborationError,
  type AdminCollaborationTransport,
  type AdminTransportRequest,
  type AdminTransportResponse,
  type JsonObject,
} from "./contracts.ts";

const COMMUNITY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057001";
const PRINCIPAL_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057002";
const INVITE_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057003";
const REPORT_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057004";
const DELETION_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057005";
const OPERATION_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057006";

type ServiceMode = "supported" | "stale" | "upgrade_required";

class ResourceService implements AdminCollaborationTransport {
  readonly requests: AdminTransportRequest[] = [];
  readonly mode: ServiceMode;

  constructor(mode: ServiceMode = "supported") {
    this.mode = mode;
  }

  async request(request: AdminTransportRequest): Promise<AdminTransportResponse> {
    this.requests.push(structuredClone(request));
    if (request.path === COMPATIBILITY_PATH) {
      const feature = (request.body?.features as unknown[] | undefined)?.[0];
      if (this.mode === "upgrade_required") {
        return {
          status: 426,
          body: {
            policy_version: 1,
            outcome: "upgrade_required",
            error: "upgrade_required",
            client_id: ADMIN_CLIENT_ID,
            minimum_client_version: "0.2.0",
            maximum_client_version: "0.2.3",
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
          selected_features: [feature],
          retryable: false,
        },
      };
    }

    if (this.mode === "stale" && request.method !== "GET") {
      return { status: 409, body: { error: "stale_write" } };
    }

    return this.resourceResponse(request);
  }

  private resourceResponse(request: AdminTransportRequest): AdminTransportResponse {
    const communityPath = `/v1/collaboration/admin/communities/${COMMUNITY_ID}`;
    if (request.path === communityPath) {
      return {
        status: 200,
        body: {
          community_id: COMMUNITY_ID,
          display_name: "Engineering",
          state: "active",
          version: 8,
        },
      };
    }
    if (request.path === `${communityPath}/members`) {
      return {
        status: 200,
        body: {
          members: [
            {
              principal_id: PRINCIPAL_ID,
              role: "admin",
              state: "active",
              version: 4,
            },
          ],
        },
      };
    }
    if (request.path === `${communityPath}/invites`) {
      return {
        status: 200,
        body: {
          invites: [
            {
              invite_id: INVITE_ID,
              role: "member",
              state: "active",
              remaining_uses: 3,
              version: 2,
            },
          ],
        },
      };
    }
    if (request.path === `${communityPath}/moderation/reports`) {
      return {
        status: 200,
        body: {
          reports: [
            {
              report_id: REPORT_ID,
              target_kind: "event",
              target_id: "ab".repeat(32),
              reason: "spam",
              state: "open",
              version: 3,
            },
          ],
        },
      };
    }
    if (request.path === `${communityPath}/deletions/${DELETION_ID}`) {
      return {
        status: 200,
        body: {
          stage: "failed",
          last_trustworthy_stage: "reversible",
          completed_phases: 0,
          total_phases: 6,
          next_phase: "database",
          halt_reason: "dependency_unavailable",
          recovery_action: "restore",
          authority_evidence: "must-not-project",
          tenant_inventory_digest: "must-not-project",
        },
      };
    }
    if (request.path === `${communityPath}/metrics`) {
      return {
        status: 200,
        body: {
          active_members: 42,
          open_reports: 2,
          pending_invites: 3,
          deletion_completed_phases: 0,
          deletion_total_phases: 6,
          measured_at_millis: 1_787_600_000_000,
        },
      };
    }
    if (request.path === `${communityPath}/archive`) {
      return {
        status: 200,
        body: {
          operation_id: OPERATION_ID,
          resource_id: COMMUNITY_ID,
          version: 9,
        },
      };
    }
    return { status: 404, body: { error: "not_found" } };
  }
}

function resourceRequests(service: ResourceService): AdminTransportRequest[] {
  return service.requests.filter((request) => request.requiredScope !== undefined);
}

function compatibilityBodies(service: ResourceService): JsonObject[] {
  return service.requests.filter((request) => request.path === COMPATIBILITY_PATH).map((request) => request.body ?? {});
}

test("resources negotiate independently and request only their canonical scope", async () => {
  const service = new ResourceService();
  const client = new AdminCollaborationClient(service);

  assert.equal((await client.getCommunity(COMMUNITY_ID)).version, 8);
  assert.equal((await client.listMembers(COMMUNITY_ID))[0]?.principalId, PRINCIPAL_ID);
  assert.equal((await client.listInvites(COMMUNITY_ID))[0]?.inviteId, INVITE_ID);
  assert.equal((await client.listModerationReports(COMMUNITY_ID))[0]?.reportId, REPORT_ID);
  const deletion = await client.getDeletionStatus(COMMUNITY_ID, DELETION_ID);
  assert.deepEqual(deletion, {
    stage: "failed",
    lastTrustworthyStage: "reversible",
    completedPhases: 0,
    totalPhases: 6,
    nextPhase: "database",
    checkpointVersion: undefined,
    haltReason: "dependency_unavailable",
    recoveryAction: "restore",
  });
  assert.equal((await client.getMetrics(COMMUNITY_ID)).activeMembers, 42);
  assert.equal((await client.archiveCommunity(COMMUNITY_ID, 8, OPERATION_ID)).version, 9);

  assert.deepEqual(
    resourceRequests(service).map((request) => request.requiredScope),
    [
      "communities:manage",
      "communities:manage",
      "communities:manage",
      "moderation:manage",
      "communities:manage",
      "communities:manage",
      "communities:manage",
    ],
  );
  assert.deepEqual(
    compatibilityBodies(service).map((body) => [body.access, body.features]),
    [
      ["read", ["admin-lifecycle"]],
      ["read", ["admin-lifecycle"]],
      ["read", ["admin-lifecycle"]],
      ["read", ["moderation"]],
      ["read", ["admin-lifecycle"]],
      ["read", ["admin-lifecycle"]],
      ["write", ["admin-lifecycle"]],
    ],
  );
});

test("stale writes retain the exact optimistic version and operation id", async () => {
  const service = new ResourceService("stale");
  const client = new AdminCollaborationClient(service);

  await assert.rejects(
    client.updateMemberRole(COMMUNITY_ID, PRINCIPAL_ID, "member", 4, OPERATION_ID),
    (error: unknown) => error instanceof AdminCollaborationError && error.kind === "stale_write",
  );
  const write = resourceRequests(service)[0];
  assert.equal(write?.requiredScope, "communities:manage");
  assert.deepEqual(write?.body, {
    role: "member",
    expected_version: 4,
    operation_id: OPERATION_ID,
  });
});

test("minimum-version errors stop before authorization and resource lookup", async () => {
  const service = new ResourceService("upgrade_required");
  const client = new AdminCollaborationClient(service);

  await assert.rejects(
    client.resolveModerationReport(COMMUNITY_ID, REPORT_ID, 3, "actioned", OPERATION_ID),
    (error: unknown) => {
      assert.ok(error instanceof AdminCollaborationError);
      assert.equal(error.kind, "upgrade_required");
      assert.equal(error.minimumVersion, "0.2.0");
      assert.equal(error.maximumVersion, "0.2.3");
      return true;
    },
  );
  assert.equal(service.requests.length, 1);
  assert.equal(service.requests[0]?.requiredScope, undefined);
  assert.equal(service.requests[0]?.body?.client_id, ADMIN_CLIENT_ID);
  assert.equal(service.requests[0]?.body?.access, "write");
});
