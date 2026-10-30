import type {
  Nip07Provider,
  SignedNostrEvent,
  UnsignedNostrEvent,
} from "../auth/nip98.ts";
import { RepositoryBrowserClient } from "./browser_client.ts";
import {
  REPOSITORIES_ROUTE,
  RepositoryBrowserError,
  parseRepositoryRoute,
  repositoryBlobPath,
  repositoryDetailPath,
  repositoryDownloadPath,
} from "./contracts.ts";

const BASE_URL = "https://collaboration.example.test";
const REPOSITORY_ID = "018fbe5f-6f37-7b40-8fb3-1c8d64057002";
const tests: Array<{ name: string; run: () => Promise<void> }> = [];

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
  kind: RepositoryBrowserError["kind"],
): Promise<void> {
  try {
    await promise;
  } catch (error) {
    assertOk(error instanceof RepositoryBrowserError);
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

function supportedCompatibility(): Record<string, unknown> {
  return {
    policy_version: 1,
    outcome: "supported",
    client_id: "buzz-web",
    minimum_client_version: "0.1.0",
    maximum_client_version: "0.1.0",
    selected_features: ["repository-browse"],
    retryable: false,
  };
}

function repository(visibility: "public" | "private") {
  return {
    repository_id: REPOSITORY_ID,
    route_id: "project-alpha",
    name: "Project Alpha",
    description: "Canonical hosted repository",
    visibility,
    default_ref: "main",
    updated_at_millis: 1_782_345_678_000,
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

function decodeAuthorization(value: string | null): SignedNostrEvent {
  assertOk(value !== null);
  assertOk(value.startsWith("Nostr "));
  const bytes = Uint8Array.from(
    atob(value.slice("Nostr ".length)),
    (character) => character.charCodeAt(0),
  );
  return JSON.parse(new TextDecoder().decode(bytes)) as SignedNostrEvent;
}

async function toRequest(
  input: string | URL | Request,
  init?: RequestInit,
): Promise<Request> {
  return input instanceof Request && init === undefined
    ? input
    : new Request(input, init);
}

it("browses public and private repositories through one canonical contract", async () => {
  const requests: Request[] = [];
  const fetchMock: typeof fetch = async (input, init) => {
    const request = await toRequest(input, init);
    requests.push(request.clone());
    const path = new URL(request.url).pathname;
    if (path === "/v1/collaboration/compatibility") {
      return jsonResponse(supportedCompatibility());
    }
    if (path === "/v1/collaboration/repositories") {
      return jsonResponse({ repositories: [repository("public")] });
    }
    if (path === "/v1/collaboration/repositories/by-route/project-alpha") {
      return jsonResponse(repository("private"));
    }
    return jsonResponse({ error: "not_found" }, 404);
  };
  const client = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: fetchMock,
    signer: approvingSigner(),
  });

  const publicRepositories = await client.listRepositories("public");
  assertEqual(publicRepositories[0]?.visibility, "public");
  const privateRepository = await client.getRepository(
    "project-alpha",
    "private",
  );
  assertEqual(privateRepository.visibility, "private");

  const listRequest = requests.find(
    (request) =>
      new URL(request.url).pathname === "/v1/collaboration/repositories",
  );
  assertOk(listRequest);
  assertEqual(listRequest.headers.get("Authorization"), null);
  const privateRequest = requests.find((request) =>
    new URL(request.url).pathname.endsWith("/by-route/project-alpha"),
  );
  assertOk(privateRequest);
  const authEvent = decodeAuthorization(
    privateRequest.headers.get("Authorization"),
  );
  assertDeepEqual(authEvent.tags, [
    ["u", `${BASE_URL}/v1/collaboration/repositories/by-route/project-alpha`],
    ["method", "GET"],
  ]);
});

it("reads an authorized blob and validates its canonical object identity", async () => {
  const bytes = new TextEncoder().encode("hello");
  const fetchMock: typeof fetch = async (input) => {
    const path = new URL(
      input instanceof Request ? input.url : input.toString(),
    ).pathname;
    if (path === "/v1/collaboration/compatibility") {
      return jsonResponse(supportedCompatibility());
    }
    return new Response(bytes, {
      status: 200,
      headers: {
        "Content-Length": String(bytes.length),
        "Content-Type": "text/plain; charset=utf-8",
        ETag: `"${"1a".repeat(20)}"`,
      },
    });
  };
  const client = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: fetchMock,
  });

  const blob = await client.readBlob(
    REPOSITORY_ID,
    "refs/heads/main",
    "docs/readme.txt",
  );
  assertEqual(new TextDecoder().decode(blob.bytes), "hello");
  assertEqual(blob.objectId, "1a".repeat(20));
  assertEqual(blob.contentType, "text/plain; charset=utf-8");
});

it("requests and verifies one bounded download range", async () => {
  let rangeHeader: string | null = null;
  const fetchMock: typeof fetch = async (input, init) => {
    const request = await toRequest(input, init);
    const path = new URL(request.url).pathname;
    if (path === "/v1/collaboration/compatibility") {
      return jsonResponse(supportedCompatibility());
    }
    rangeHeader = request.headers.get("Range");
    return new Response(Uint8Array.from([2, 3, 4, 5]), {
      status: 206,
      headers: {
        "Content-Length": "4",
        "Content-Range": "bytes 2-5/10",
        "Content-Type": "application/octet-stream",
        ETag: `"${"2b".repeat(20)}"`,
      },
    });
  };
  const client = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: fetchMock,
  });

  const download = await client.downloadBlob(
    REPOSITORY_ID,
    "main",
    "artifacts/archive.bin",
    { range: { start: 2, end: 5 } },
  );
  assertEqual(rangeHeader, "bytes=2-5");
  assertEqual(download.status, 206);
  assertDeepEqual(download.range, { start: 2, end: 5, total: 10 });
  assertDeepEqual([...download.bytes], [2, 3, 4, 5]);
});

it("rejects a partial response for a different byte range", async () => {
  const fetchMock: typeof fetch = async (input) => {
    const path = new URL(
      input instanceof Request ? input.url : input.toString(),
    ).pathname;
    if (path === "/v1/collaboration/compatibility") {
      return jsonResponse(supportedCompatibility());
    }
    return new Response(Uint8Array.from([3, 4, 5, 6]), {
      status: 206,
      headers: {
        "Content-Length": "4",
        "Content-Range": "bytes 3-6/10",
        ETag: `"${"2b".repeat(20)}"`,
      },
    });
  };
  const client = new RepositoryBrowserClient({
    baseUrl: BASE_URL,
    fetch: fetchMock,
  });

  await assertRejects(
    client.downloadBlob(REPOSITORY_ID, "main", "artifacts/archive.bin", {
      range: { start: 2, end: 5 },
    }),
    "invalid_response",
  );
});

it("keeps missing and denied objects closed and preserves old web URLs", async () => {
  for (const status of [403, 404]) {
    const fetchMock: typeof fetch = async (input) => {
      const path = new URL(
        input instanceof Request ? input.url : input.toString(),
      ).pathname;
      return path === "/v1/collaboration/compatibility"
        ? jsonResponse(supportedCompatibility())
        : jsonResponse({ error: "unavailable" }, status);
    };
    const client = new RepositoryBrowserClient({
      baseUrl: BASE_URL,
      fetch: fetchMock,
      signer: approvingSigner(),
    });
    await assertRejects(
      client.readBlob(REPOSITORY_ID, "main", "missing.txt", "private"),
      "object_unavailable",
    );
  }

  assertEqual(REPOSITORIES_ROUTE, "/repos");
  assertEqual(repositoryDetailPath("project-alpha"), "/repos/project-alpha");
  assertEqual(
    repositoryBlobPath("project-alpha", "docs/guide.md"),
    "/repos/project-alpha/blob/docs/guide.md",
  );
  assertEqual(
    repositoryDownloadPath("project-alpha", "docs/guide.md", "main"),
    "/repos/project-alpha/blob/docs/guide.md?download=1&ref=main",
  );
  assertDeepEqual(
    parseRepositoryRoute(
      "/repos/project-alpha/blob/docs/guide.md?download=1&ref=main",
    ),
    {
      kind: "blob",
      routeId: "project-alpha",
      path: "docs/guide.md",
      ref: "main",
      download: true,
    },
  );
});

for (const test of tests) {
  await test.run();
  console.log(`ok - ${test.name}`);
}
