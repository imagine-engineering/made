import {
  WEB_CLIENT_ID,
  WEB_CLIENT_VERSION,
  CollaborationAuthError,
  buzzInviteDeepLink,
  invitePagePath,
  type InvitePolicyAcceptance,
} from "../../src/auth/contracts.ts";
import { CollaborationWebAuthClient } from "../../src/auth/invite_flow.ts";
import type {
  Nip07Provider,
  SignedNostrEvent,
  UnsignedNostrEvent,
} from "../../src/auth/nip98.ts";
import { RepositoryBrowserClient } from "../../src/repositories/browser_client.ts";
import {
  REPOSITORIES_ROUTE,
  RepositoryBrowserError,
  repositoryBlobPath,
  repositoryDetailPath,
  repositoryDownloadPath,
} from "../../src/repositories/contracts.ts";

const BASE_URL = "https://collaboration.example.test";
const COMMUNITY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057001";
const REPOSITORY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057002";
const FROZEN_WEB_VERSION = "0.1.0";
const FROZEN_ROUTES = [
  "/",
  "/invite/$code",
  "/repos",
  "/repos/$repoId",
  "/repos/$repoId/blob/$",
] as const;
const tests: Array<{ name: string; run: () => Promise<void> }> = [];

type CompatibilityMode = "supported" | "read_only" | "upgrade_required";
type JsonObject = Record<string, unknown>;

function it(name: string, run: () => Promise<void>): void {
  tests.push({ name, run });
}

function assertOk(value: unknown, message = "assertion failed"): asserts value {
  if (!value) throw new Error(message);
}

function assertEqual(actual: unknown, expected: unknown): void {
  if (!Object.is(actual, expected)) {
    throw new Error(`expected ${String(expected)}, received ${String(actual)}`);
  }
}

function assertDeepEqual(actual: unknown, expected: unknown): void {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
    );
  }
}

async function assertRejects(
  promise: Promise<unknown>,
  errorType: typeof CollaborationAuthError | typeof RepositoryBrowserError,
  kind: string,
): Promise<void> {
  try {
    await promise;
  } catch (error) {
    assertOk(error instanceof errorType);
    assertEqual(error.kind, kind);
    return;
  }
  throw new Error("expected promise to reject");
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function requestFrom(
  input: string | URL | Request,
  init?: RequestInit,
): Promise<Request> {
  return input instanceof Request && init === undefined
    ? input
    : new Request(input, init);
}

class CompatibilityService {
  readonly mode: CompatibilityMode;
  readonly requests: Request[] = [];
  readonly compatibilityBodies: JsonObject[] = [];
  readonly resourcePaths: string[] = [];
  mutations = 0;

  constructor(mode: CompatibilityMode) {
    this.mode = mode;
  }

  readonly fetch: typeof fetch = async (input, init) => {
    const request = await requestFrom(input, init);
    this.requests.push(request.clone());
    const url = new URL(request.url);
    if (url.pathname === "/v1/collaboration/compatibility") {
      const body = (await request.json()) as JsonObject;
      this.compatibilityBodies.push(body);
      if (this.mode === "upgrade_required") {
        return jsonResponse(
          {
            policy_version: 1,
            outcome: "upgrade_required",
            error: "upgrade_required",
            client_id: WEB_CLIENT_ID,
            minimum_client_version: "0.2.0",
            maximum_client_version: "0.2.4",
            selected_features: [],
            retryable: false,
          },
          426,
        );
      }
      const access = body.access;
      const outcome =
        this.mode === "read_only" && access === "write"
          ? "read_only"
          : "supported";
      return jsonResponse({
        policy_version: 1,
        outcome,
        client_id: WEB_CLIENT_ID,
        minimum_client_version: FROZEN_WEB_VERSION,
        maximum_client_version: FROZEN_WEB_VERSION,
        selected_features: body.features,
        retryable: false,
      });
    }

    this.resourcePaths.push(`${request.method} ${url.pathname}`);
    if (url.pathname === "/v1/collaboration/invites/resolve") {
      return jsonResponse({
        community_id: COMMUNITY_ID,
        community_host: "relay.example.com",
        status: "active",
        role: "member",
      });
    }
    if (
      url.pathname ===
      `/v1/collaboration/communities/${COMMUNITY_ID}/join-policy`
    ) {
      return jsonResponse({
        policy: {
          version: "policy-v1",
          terms_markdown: "Terms",
          privacy_markdown: "Privacy",
          age_attestation_required: true,
        },
      });
    }
    if (url.pathname === "/v1/collaboration/invites/redeem") {
      this.mutations += 1;
      return jsonResponse({
        status: "joined",
        community_id: COMMUNITY_ID,
        community_host: "relay.example.com",
        role: "member",
      });
    }
    if (url.pathname === "/v1/collaboration/repositories") {
      return jsonResponse({
        repositories: [
          {
            repository_id: REPOSITORY_ID,
            route_id: "project-alpha",
            name: "Project Alpha",
            description: "Canonical hosted repository",
            visibility: "public",
            default_ref: "main",
            updated_at_millis: 1_782_345_678_000,
          },
        ],
      });
    }
    if (
      url.pathname === `/v1/collaboration/repositories/${REPOSITORY_ID}/blob`
    ) {
      const bytes = new TextEncoder().encode("hello");
      return new Response(bytes, {
        status: 200,
        headers: {
          "Content-Length": String(bytes.length),
          "Content-Type": "text/plain",
          ETag: `"${"1a".repeat(20)}"`,
        },
      });
    }
    return jsonResponse({ error: "not_found" }, 404);
  };
}

class CountingSigner implements Nip07Provider {
  calls = 0;
  readonly publicKey = "ab".repeat(32);

  async getPublicKey(): Promise<string> {
    return this.publicKey;
  }

  async signEvent(event: UnsignedNostrEvent): Promise<SignedNostrEvent> {
    this.calls += 1;
    return {
      ...event,
      id: "cd".repeat(32),
      pubkey: this.publicKey,
      sig: "ef".repeat(64),
    };
  }
}

function decodeAuthorization(request: Request): SignedNostrEvent {
  const authorization = request.headers.get("Authorization");
  assertOk(authorization !== null);
  assertOk(authorization.startsWith("Nostr "));
  const bytes = Uint8Array.from(
    atob(authorization.slice("Nostr ".length)),
    (character) => character.charCodeAt(0),
  );
  return JSON.parse(new TextDecoder().decode(bytes)) as SignedNostrEvent;
}

it("runs the frozen and migrated supported web version end to end", async () => {
  assertEqual(WEB_CLIENT_VERSION, FROZEN_WEB_VERSION);
  assertDeepEqual(FROZEN_ROUTES, [
    "/",
    "/invite/$code",
    REPOSITORIES_ROUTE,
    "/repos/$repoId",
    "/repos/$repoId/blob/$",
  ]);

  const service = new CompatibilityService("supported");
  const signer = new CountingSigner();
  const auth = new CollaborationWebAuthClient({
    baseUrl: BASE_URL,
    fetch: service.fetch,
    signer,
  });
  const repositories = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: service.fetch,
  });

  assertEqual(invitePagePath("demo-code"), "/invite/demo-code");
  const invite = await auth.loadInvite("demo-code");
  assertEqual(invite.joinPolicy?.version, "policy-v1");
  assertEqual(invite.joinPolicy?.age_attestation_required, true);
  assertOk(invite.joinPolicy?.terms_markdown);
  assertOk(invite.joinPolicy?.privacy_markdown);

  const acceptance: InvitePolicyAcceptance = {
    policyVersion: "policy-v1",
    ageConfirmed: true,
    legalDocumentsAccepted: true,
  };
  const redemption = await auth.redeemInvite("demo-code", acceptance);
  assertEqual(redemption.status, "joined");
  assertEqual(signer.calls, 1);
  assertEqual(service.mutations, 1);

  const listed = await repositories.listRepositories();
  assertEqual(listed[0]?.routeId, "project-alpha");
  const blob = await repositories.readBlob(
    REPOSITORY_ID,
    "main",
    "docs/readme.txt",
  );
  assertEqual(new TextDecoder().decode(blob.bytes), "hello");
  assertEqual(blob.objectId, "1a".repeat(20));

  assertEqual(repositoryDetailPath("project-alpha"), "/repos/project-alpha");
  assertEqual(
    repositoryBlobPath("project-alpha", "docs/readme.txt"),
    "/repos/project-alpha/blob/docs/readme.txt",
  );
  assertEqual(
    repositoryDownloadPath("project-alpha", "docs/readme.txt", "main"),
    "/repos/project-alpha/blob/docs/readme.txt?download=1&ref=main",
  );
  assertEqual(
    buzzInviteDeepLink("wss://relay.example.com", "abc123", "receipt.value"),
    "buzz://join?relay=wss%3A%2F%2Frelay.example.com&code=abc123&policy_receipt=receipt.value",
  );

  assertOk(
    service.compatibilityBodies.every(
      (body) =>
        body.client_id === WEB_CLIENT_ID &&
        body.client_version === FROZEN_WEB_VERSION,
    ),
  );
  const redeemRequest = service.requests.find(
    (request) =>
      new URL(request.url).pathname === "/v1/collaboration/invites/redeem",
  );
  assertOk(redeemRequest);
  const event = decodeAuthorization(redeemRequest);
  assertDeepEqual(
    event.tags.map((tag) => tag[0]),
    ["u", "method", "payload", "nonce"],
  );
});

it("rejects an incompatible write before signer, tenant access, or mutation", async () => {
  const service = new CompatibilityService("upgrade_required");
  const signer = new CountingSigner();
  const auth = new CollaborationWebAuthClient({
    baseUrl: BASE_URL,
    fetch: service.fetch,
    signer,
  });

  await assertRejects(
    auth.redeemInvite("demo-code"),
    CollaborationAuthError,
    "upgrade_required",
  );
  assertEqual(signer.calls, 0);
  assertEqual(service.resourcePaths.length, 0);
  assertEqual(service.mutations, 0);
  assertEqual(service.compatibilityBodies.length, 1);
});

it("keeps a read-only client from signing or mutating", async () => {
  const service = new CompatibilityService("read_only");
  const signer = new CountingSigner();
  const auth = new CollaborationWebAuthClient({
    baseUrl: BASE_URL,
    fetch: service.fetch,
    signer,
  });

  await assertRejects(
    auth.redeemInvite("demo-code"),
    CollaborationAuthError,
    "read_only",
  );
  assertEqual(signer.calls, 0);
  assertEqual(service.resourcePaths.length, 0);
  assertEqual(service.mutations, 0);
});

it("rejects incompatible repository reads before object lookup", async () => {
  const service = new CompatibilityService("upgrade_required");
  const repositories = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: service.fetch,
  });

  await assertRejects(
    repositories.downloadBlob(REPOSITORY_ID, "main", "docs/readme.txt", {
      range: { start: 0, end: 4 },
    }),
    RepositoryBrowserError,
    "upgrade_required",
  );
  assertEqual(service.resourcePaths.length, 0);
  assertEqual(service.mutations, 0);
});

for (const test of tests) {
  await test.run();
  console.log(`ok - ${test.name}`);
}
