import {
  COMPATIBILITY_PATH,
  MINIMUM_COMPATIBILITY_POLICY_VERSION,
  WEB_CLIENT_ID,
  WEB_CLIENT_VERSION,
  CollaborationAuthError,
  type CompatibilityResponse,
} from "../auth/contracts.ts";
import { makeNip98Authorization, type Nip07Provider } from "../auth/nip98.ts";
import {
  MAX_BLOB_PREVIEW_BYTES,
  MAX_DOWNLOAD_RANGE_BYTES,
  REPOSITORIES_API_PATH,
  RepositoryBrowserError,
  repositoryBlobApiUrl,
  repositoryTreeApiUrl,
  type RepositoryAccess,
  type RepositoryBlob,
  type RepositoryDownload,
  type RepositorySummary,
  type RepositoryTreeEntry,
  validBlobPath,
  validGitRef,
  validRepositoryId,
  validRouteId,
} from "./contracts.ts";

const REQUEST_TIMEOUT_MS = 15_000;
const MAX_JSON_RESPONSE_CHARACTERS = 1024 * 1024;
const MAX_REPOSITORIES = 1_000;
const MAX_TREE_ENTRIES = 10_000;

type JsonObject = Record<string, unknown>;

export type RepositoryBrowserClientOptions = {
  baseUrl: string;
  fetch?: typeof fetch;
  signer?: Nip07Provider;
  timeoutMilliseconds?: number;
};

export class RepositoryBrowserClient {
  readonly #baseUrl: URL;
  readonly #fetch: typeof fetch;
  readonly #signer: Nip07Provider | undefined;
  readonly #timeoutMilliseconds: number;

  constructor(options: RepositoryBrowserClientOptions) {
    this.#baseUrl = serviceBaseUrl(options.baseUrl);
    this.#fetch = options.fetch ?? fetch;
    this.#signer = options.signer;
    const timeout = options.timeoutMilliseconds ?? REQUEST_TIMEOUT_MS;
    if (!Number.isInteger(timeout) || timeout < 1 || timeout > 60_000) {
      throw invalidRequest();
    }
    this.#timeoutMilliseconds = timeout;
  }

  async listRepositories(
    access: RepositoryAccess = "public",
  ): Promise<RepositorySummary[]> {
    await this.#negotiate();
    const response = await this.#getJson(
      new URL(REPOSITORIES_API_PATH, this.#baseUrl),
      access,
      "repository",
    );
    if (
      !Array.isArray(response.repositories) ||
      response.repositories.length > MAX_REPOSITORIES
    ) {
      throw invalidResponse();
    }
    return response.repositories.map(parseRepository);
  }

  async getRepository(
    routeId: string,
    access: RepositoryAccess = "public",
  ): Promise<RepositorySummary> {
    await this.#negotiate();
    const url = new URL(
      `${REPOSITORIES_API_PATH}/by-route/${encodeURIComponent(validRouteId(routeId))}`,
      this.#baseUrl,
    );
    return parseRepository(await this.#getJson(url, access, "repository"));
  }

  async browseTree(
    repositoryId: string,
    ref: string,
    path: string | undefined,
    access: RepositoryAccess = "public",
  ): Promise<RepositoryTreeEntry[]> {
    await this.#negotiate();
    const response = await this.#getJson(
      repositoryTreeApiUrl(this.#baseUrl, repositoryId, ref, path),
      access,
      "object",
    );
    if (
      !Array.isArray(response.entries) ||
      response.entries.length > MAX_TREE_ENTRIES
    ) {
      throw invalidResponse();
    }
    return response.entries.map(parseTreeEntry);
  }

  async readBlob(
    repositoryId: string,
    ref: string,
    path: string,
    access: RepositoryAccess = "public",
  ): Promise<RepositoryBlob> {
    await this.#negotiate();
    const response = await this.#get(
      repositoryBlobApiUrl(this.#baseUrl, repositoryId, ref, path),
      access,
    );
    if (!response.ok) throw httpError(response.status, "object");
    return readBlobResponse(response, MAX_BLOB_PREVIEW_BYTES);
  }

  async downloadBlob(
    repositoryId: string,
    ref: string,
    path: string,
    options?: {
      access?: RepositoryAccess;
      range?: { start: number; end: number };
    },
  ): Promise<RepositoryDownload> {
    await this.#negotiate();
    const headers = new Headers();
    if (options?.range) {
      const { start, end } = options.range;
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) ||
        start < 0 ||
        end < start ||
        end - start + 1 > MAX_DOWNLOAD_RANGE_BYTES
      ) {
        throw invalidRequest();
      }
      headers.set("Range", `bytes=${start}-${end}`);
    }
    const response = await this.#get(
      repositoryBlobApiUrl(this.#baseUrl, repositoryId, ref, path),
      options?.access ?? "public",
      headers,
    );
    if (!response.ok) throw httpError(response.status, "object");
    if (options?.range && response.status !== 206) throw invalidResponse();
    if (!options?.range && response.status !== 200) throw invalidResponse();

    const blob = await readBlobResponse(response, MAX_DOWNLOAD_RANGE_BYTES);
    const range = options?.range
      ? parseContentRange(
          response.headers.get("Content-Range"),
          blob.bytes.length,
          options.range,
        )
      : undefined;
    return {
      ...blob,
      status: response.status as 200 | 206,
      range,
    };
  }

  async #negotiate(): Promise<void> {
    const response = await this.#request(
      new URL(COMPATIBILITY_PATH, this.#baseUrl),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: WEB_CLIENT_ID,
          client_version: WEB_CLIENT_VERSION,
          access: "read",
          protocols: [{ id: "collaboration-http", version: 1 }],
          features: ["repository-browse"],
        }),
        redirect: "error",
      },
    );
    const json = await readJsonObject(response);
    const compatibility = json as unknown as CompatibilityResponse;
    if (!response.ok || compatibility.outcome === "upgrade_required") {
      if (
        response.status === 426 ||
        compatibility.error === "upgrade_required"
      ) {
        const minimum = optionalString(compatibility.minimum_client_version);
        const maximum = optionalString(compatibility.maximum_client_version);
        throw new RepositoryBrowserError(
          "upgrade_required",
          versionMessage(minimum, maximum),
          { minimumVersion: minimum, maximumVersion: maximum },
        );
      }
      throw serviceUnavailable(response.status >= 500);
    }
    if (
      !Number.isInteger(compatibility.policy_version) ||
      compatibility.policy_version < MINIMUM_COMPATIBILITY_POLICY_VERSION ||
      compatibility.outcome !== "supported" ||
      compatibility.client_id !== WEB_CLIENT_ID ||
      compatibility.retryable !== false ||
      !Array.isArray(compatibility.selected_features) ||
      !compatibility.selected_features.includes("repository-browse")
    ) {
      throw invalidResponse();
    }
  }

  async #getJson(
    url: URL,
    access: RepositoryAccess,
    missingKind: "repository" | "object",
  ): Promise<JsonObject> {
    const response = await this.#get(url, access);
    if (!response.ok) throw httpError(response.status, missingKind);
    return readJsonObject(response);
  }

  async #get(
    url: URL,
    access: RepositoryAccess,
    headers = new Headers(),
  ): Promise<Response> {
    if (access === "private") {
      try {
        headers.set(
          "Authorization",
          await makeNip98Authorization(this.#signer, url.toString(), "GET"),
        );
      } catch (error) {
        if (error instanceof CollaborationAuthError) {
          throw new RepositoryBrowserError(
            "authentication_denied",
            "A valid NIP-07 identity is required to browse this repository.",
          );
        }
        throw error;
      }
    }
    return this.#request(url, {
      method: "GET",
      headers,
      redirect: "error",
    });
  }

  async #request(url: URL, init: RequestInit): Promise<Response> {
    try {
      return await this.#fetch(url, {
        ...init,
        signal: AbortSignal.timeout(this.#timeoutMilliseconds),
      });
    } catch (error) {
      if (error instanceof RepositoryBrowserError) throw error;
      throw serviceUnavailable(true);
    }
  }
}

async function readJsonObject(response: Response): Promise<JsonObject> {
  const text = await response.text();
  if (text.length > MAX_JSON_RESPONSE_CHARACTERS) throw invalidResponse();
  try {
    const value: unknown = JSON.parse(text);
    if (value === null || Array.isArray(value) || typeof value !== "object") {
      throw invalidResponse();
    }
    return value as JsonObject;
  } catch (error) {
    if (error instanceof RepositoryBrowserError) throw error;
    throw invalidResponse();
  }
}

async function readBlobResponse(
  response: Response,
  maximumBytes: number,
): Promise<RepositoryBlob> {
  const declaredLength = response.headers.get("Content-Length");
  if (declaredLength !== null) {
    const length = Number(declaredLength);
    if (!Number.isSafeInteger(length) || length < 0 || length > maximumBytes) {
      throw invalidResponse();
    }
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (
    bytes.length > maximumBytes ||
    (declaredLength !== null && Number(declaredLength) !== bytes.length)
  ) {
    throw invalidResponse();
  }
  const objectId = response.headers.get("ETag")?.replace(/^"|"$/g, "");
  if (
    objectId === undefined ||
    !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(objectId)
  ) {
    throw invalidResponse();
  }
  return {
    bytes,
    contentType:
      response.headers.get("Content-Type") ?? "application/octet-stream",
    objectId,
  };
}

function parseRepository(value: unknown): RepositorySummary {
  const object = asObject(value);
  const visibility = object.visibility;
  if (visibility !== "public" && visibility !== "private")
    throw invalidResponse();
  const updatedAtMillis = object.updated_at_millis;
  if (!Number.isSafeInteger(updatedAtMillis) || Number(updatedAtMillis) <= 0) {
    throw invalidResponse();
  }
  return {
    repositoryId: validResponseValue(
      requiredString(object, "repository_id"),
      validRepositoryId,
    ),
    routeId: validResponseValue(
      requiredString(object, "route_id"),
      validRouteId,
    ),
    name: boundedString(object, "name", 128),
    description: boundedString(object, "description", 1_024, true),
    visibility,
    defaultRef: validResponseValue(
      requiredString(object, "default_ref"),
      validGitRef,
    ),
    updatedAtMillis: Number(updatedAtMillis),
  };
}

function parseTreeEntry(value: unknown): RepositoryTreeEntry {
  const object = asObject(value);
  const kind = object.kind;
  if (kind !== "blob" && kind !== "tree") throw invalidResponse();
  const objectId = requiredString(object, "object_id");
  if (!/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(objectId))
    throw invalidResponse();
  const size = object.size_bytes;
  if (size !== undefined && (!Number.isSafeInteger(size) || Number(size) < 0)) {
    throw invalidResponse();
  }
  const name = boundedString(object, "name", 255);
  if (name === "." || name === ".." || name.includes("/"))
    throw invalidResponse();
  return {
    name,
    path: validResponseValue(
      boundedString(object, "path", 4_096),
      validBlobPath,
    ),
    kind,
    objectId,
    sizeBytes: size === undefined ? undefined : Number(size),
  };
}

function parseContentRange(
  value: string | null,
  byteLength: number,
  requested: { start: number; end: number },
) {
  const match = value?.match(/^bytes (\d+)-(\d+)\/(\d+)$/);
  if (!match) throw invalidResponse();
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = Number(match[3]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    !Number.isSafeInteger(total) ||
    start < 0 ||
    end < start ||
    end >= total ||
    start !== requested.start ||
    end !== requested.end ||
    end - start + 1 !== byteLength
  ) {
    throw invalidResponse();
  }
  return { start, end, total };
}

function validResponseValue(
  value: string,
  validate: (value: string) => string,
): string {
  try {
    return validate(value);
  } catch {
    throw invalidResponse();
  }
}

function asObject(value: unknown): JsonObject {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw invalidResponse();
  }
  return value as JsonObject;
}

function requiredString(object: JsonObject, key: string): string {
  return boundedString(object, key, 4_096);
}

function boundedString(
  object: JsonObject,
  key: string,
  maximum: number,
  empty = false,
): string {
  const value = object[key];
  if (
    typeof value !== "string" ||
    (!empty && value.length === 0) ||
    value.length > maximum
  ) {
    throw invalidResponse();
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 && value.length <= 128
    ? value
    : undefined;
}

function serviceBaseUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw invalidRequest();
  }
  const localHttp =
    url.protocol === "http:" &&
    (url.hostname === "localhost" ||
      url.hostname.endsWith(".localhost") ||
      url.hostname === "127.0.0.1" ||
      url.hostname === "[::1]");
  if (
    (url.protocol !== "https:" && !localHttp) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw invalidRequest();
  }
  url.pathname = "/";
  return url;
}

function httpError(
  status: number,
  missingKind: "repository" | "object",
): RepositoryBrowserError {
  if (status === 404 || status === 403) {
    return new RepositoryBrowserError(
      missingKind === "repository"
        ? "repository_unavailable"
        : "object_unavailable",
      missingKind === "repository"
        ? "This repository is unavailable."
        : "This repository object is unavailable.",
    );
  }
  return serviceUnavailable(status >= 500);
}

function versionMessage(minimum?: string, maximum?: string): string {
  if (minimum && maximum) {
    return `Buzz web ${WEB_CLIENT_VERSION} is unsupported. Use version ${minimum} through ${maximum}.`;
  }
  return "This Buzz web version is unsupported. Upgrade before continuing.";
}

function invalidRequest(): RepositoryBrowserError {
  return new RepositoryBrowserError(
    "invalid_request",
    "The repository request is invalid.",
  );
}

function invalidResponse(): RepositoryBrowserError {
  return new RepositoryBrowserError(
    "invalid_response",
    "The repository service returned an invalid response.",
  );
}

function serviceUnavailable(retryable: boolean): RepositoryBrowserError {
  return new RepositoryBrowserError(
    "service_unavailable",
    "The repository service is unavailable.",
    { retryable },
  );
}
