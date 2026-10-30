import {
  buzzInviteDeepLink,
  CollaborationAuthError,
  invitePagePath,
} from "./contracts.ts";
import { CollaborationWebAuthClient } from "./invite_flow.ts";
import type {
  Nip07Provider,
  SignedNostrEvent,
  UnsignedNostrEvent,
} from "./nip98.ts";

const BASE_URL = "https://collaboration.example.test";
const COMMUNITY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057001";
const tests: Array<{ name: string; run: () => Promise<void> }> = [];

function describe(_name: string, register: () => void): void {
  register();
}

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

function assertMatch(actual: string, expected: RegExp): void {
  if (!expected.test(actual)) {
    throw new Error(`expected ${JSON.stringify(actual)} to match ${expected}`);
  }
}

async function assertRejects(
  promise: Promise<unknown>,
  verify: (error: unknown) => boolean,
): Promise<void> {
  try {
    await promise;
  } catch (error) {
    if (verify(error)) return;
    throw new Error("rejection verifier returned false");
  }
  throw new Error("expected promise to reject");
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function supportedCompatibility() {
  return {
    policy_version: 1,
    outcome: "supported",
    client_id: "buzz-web",
    minimum_client_version: "0.1.0",
    maximum_client_version: "0.1.0",
    selected_features: ["communities", "invites"],
    retryable: false,
  };
}

function approvingSigner(): Nip07Provider {
  const publicKey = "ab".repeat(32);
  return {
    async getPublicKey() {
      return publicKey;
    },
    async signEvent(event: UnsignedNostrEvent): Promise<SignedNostrEvent> {
      return {
        ...event,
        id: "cd".repeat(32),
        pubkey: publicKey,
        sig: "ef".repeat(64),
      };
    },
  };
}

async function requestFrom(
  input: string | URL | Request,
  init?: RequestInit,
): Promise<Request> {
  return input instanceof Request && init === undefined
    ? input
    : new Request(input, init);
}

function assertAuthError(
  error: unknown,
  kind: CollaborationAuthError["kind"],
): boolean {
  assertOk(error instanceof CollaborationAuthError);
  assertEqual(error.kind, kind);
  assertEqual(error.retryable, false);
  return true;
}

describe("canonical browser invite authentication", () => {
  it("preserves the invite route, resolves community policy, and redeems with NIP-98", async () => {
    const requests: Request[] = [];
    const fetchMock: typeof fetch = async (input, init) => {
      const request = await requestFrom(input, init);
      requests.push(request.clone());
      const path = new URL(request.url).pathname;
      if (path === "/v1/collaboration/compatibility") {
        return jsonResponse(supportedCompatibility());
      }
      if (path === "/v1/collaboration/invites/resolve") {
        return jsonResponse({
          status: "active",
          community_id: COMMUNITY_ID,
          community_host: "community.example.test",
          role: "member",
        });
      }
      if (
        path === `/v1/collaboration/communities/${COMMUNITY_ID}/join-policy`
      ) {
        return jsonResponse({
          policy: {
            version: "policy-v1",
            terms_markdown: "# Terms",
            privacy_markdown: "# Privacy",
            age_attestation_required: true,
          },
        });
      }
      if (path === "/v1/collaboration/invites/redeem") {
        return jsonResponse({
          status: "joined",
          community_id: COMMUNITY_ID,
          community_host: "community.example.test",
          role: "member",
        });
      }
      return jsonResponse({ error: "not_found" }, 404);
    };
    const client = new CollaborationWebAuthClient({
      baseUrl: BASE_URL,
      fetch: fetchMock,
      signer: approvingSigner(),
    });

    assertEqual(invitePagePath("browser-code"), "/invite/browser-code");
    assertEqual(
      buzzInviteDeepLink("wss://relay.example.com", "abc123", "receipt.value"),
      "buzz://join?relay=wss%3A%2F%2Frelay.example.com&code=abc123&policy_receipt=receipt.value",
    );
    const invite = await client.loadInvite("browser-code");
    assertEqual(invite.communityId, COMMUNITY_ID);
    assertEqual(invite.joinPolicy?.version, "policy-v1");

    const redemption = await client.redeemInvite("browser-code", {
      policyVersion: "policy-v1",
      ageConfirmed: true,
      legalDocumentsAccepted: true,
    });
    assertEqual(redemption.status, "joined");

    const redeemRequest = requests.find(
      (request) =>
        new URL(request.url).pathname === "/v1/collaboration/invites/redeem",
    );
    assertOk(redeemRequest);
    const body = await redeemRequest.clone().text();
    assertDeepEqual(JSON.parse(body), {
      code: "browser-code",
      policy_acceptance: {
        policy_version: "policy-v1",
        age_confirmed: true,
        legal_documents_accepted: true,
      },
    });
    const authorization = redeemRequest.headers.get("Authorization");
    assertMatch(authorization ?? "", /^Nostr /);
    const encodedEvent = (authorization ?? "").slice("Nostr ".length);
    const event = JSON.parse(
      new TextDecoder().decode(
        Uint8Array.from(atob(encodedEvent), (character) =>
          character.charCodeAt(0),
        ),
      ),
    ) as SignedNostrEvent;
    assertDeepEqual(event.tags.slice(0, 2), [
      ["u", `${BASE_URL}/v1/collaboration/invites/redeem`],
      ["method", "POST"],
    ]);
    assertEqual(event.tags[2]?.[0], "payload");
    const expectedPayload = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(body),
    );
    const expectedPayloadHex = Array.from(
      new Uint8Array(expectedPayload),
      (byte) => byte.toString(16).padStart(2, "0"),
    ).join("");
    assertEqual(event.tags[2]?.[1], expectedPayloadHex);
    assertEqual(event.tags[3]?.[0], "nonce");
  });

  it("renders explicit expired and exhausted invite failures", async () => {
    for (const [serverError, expectedKind] of [
      ["invite_expired", "invite_expired"],
      ["invite_exhausted", "invite_exhausted"],
    ] as const) {
      const fetchMock: typeof fetch = async (input) => {
        const path = new URL(
          input instanceof Request ? input.url : input.toString(),
        ).pathname;
        return path === "/v1/collaboration/compatibility"
          ? jsonResponse(supportedCompatibility())
          : jsonResponse({ error: serverError }, 409);
      };
      const client = new CollaborationWebAuthClient({
        baseUrl: BASE_URL,
        fetch: fetchMock,
        signer: approvingSigner(),
      });

      await assertRejects(client.redeemInvite("browser-code"), (error) =>
        assertAuthError(error, expectedKind),
      );
    }
  });

  it("stops before invite redemption when the NIP-07 signer denies", async () => {
    const requestedPaths: string[] = [];
    const fetchMock: typeof fetch = async (input) => {
      requestedPaths.push(
        new URL(input instanceof Request ? input.url : input.toString())
          .pathname,
      );
      return jsonResponse(supportedCompatibility());
    };
    const signer: Nip07Provider = {
      async getPublicKey() {
        return "ab".repeat(32);
      },
      async signEvent() {
        throw new Error("denied");
      },
    };
    const client = new CollaborationWebAuthClient({
      baseUrl: BASE_URL,
      fetch: fetchMock,
      signer,
    });

    await assertRejects(client.redeemInvite("browser-code"), (error) =>
      assertAuthError(error, "signer_denied"),
    );
    assertDeepEqual(requestedPaths, ["/v1/collaboration/compatibility"]);
  });

  it("shows the closed minimum version before signer or invite access", async () => {
    const requestedPaths: string[] = [];
    let signerCalled = false;
    const fetchMock: typeof fetch = async (input) => {
      requestedPaths.push(
        new URL(input instanceof Request ? input.url : input.toString())
          .pathname,
      );
      return jsonResponse(
        {
          policy_version: 2,
          outcome: "upgrade_required",
          error: "upgrade_required",
          reason: "client_version_unsupported",
          client_id: "buzz-web",
          minimum_client_version: "0.2.0",
          maximum_client_version: "0.2.4",
          selected_features: [],
          retryable: false,
        },
        426,
      );
    };
    const signer: Nip07Provider = {
      async getPublicKey() {
        signerCalled = true;
        return "ab".repeat(32);
      },
      async signEvent(event) {
        signerCalled = true;
        return { ...event, id: "", pubkey: "", sig: "" };
      },
    };
    const client = new CollaborationWebAuthClient({
      baseUrl: BASE_URL,
      fetch: fetchMock,
      signer,
    });

    await assertRejects(client.redeemInvite("browser-code"), (error) => {
      assertAuthError(error, "upgrade_required");
      assertOk(error instanceof CollaborationAuthError);
      assertEqual(error.minimumVersion, "0.2.0");
      assertMatch(error.message, /0\.2\.0 through 0\.2\.4/);
      return true;
    });
    assertEqual(signerCalled, false);
    assertDeepEqual(requestedPaths, ["/v1/collaboration/compatibility"]);
  });
});

for (const test of tests) {
  await test.run();
  console.log(`ok - ${test.name}`);
}
