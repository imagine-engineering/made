export const REPOSITORIES_ROUTE = "/repos";
export const REPOSITORIES_API_PATH = "/v1/collaboration/repositories";
export const MAX_BLOB_PREVIEW_BYTES = 1024 * 1024;
export const MAX_DOWNLOAD_RANGE_BYTES = 8 * 1024 * 1024;

export type RepositoryAccess = "public" | "private";

export type RepositorySummary = {
  repositoryId: string;
  routeId: string;
  name: string;
  description: string;
  visibility: RepositoryAccess;
  defaultRef: string;
  updatedAtMillis: number;
};

export type RepositoryTreeEntry = {
  name: string;
  path: string;
  kind: "blob" | "tree";
  objectId: string;
  sizeBytes?: number;
};

export type RepositoryBlob = {
  bytes: Uint8Array;
  contentType: string;
  objectId: string;
};

export type RepositoryDownload = RepositoryBlob & {
  status: 200 | 206;
  range?: { start: number; end: number; total: number };
};

export type RepositoryBrowserErrorKind =
  | "upgrade_required"
  | "authentication_denied"
  | "repository_unavailable"
  | "object_unavailable"
  | "invalid_request"
  | "invalid_response"
  | "service_unavailable";

export class RepositoryBrowserError extends Error {
  readonly kind: RepositoryBrowserErrorKind;
  readonly minimumVersion?: string;
  readonly maximumVersion?: string;
  readonly retryable: boolean;

  constructor(
    kind: RepositoryBrowserErrorKind,
    message: string,
    options?: {
      minimumVersion?: string;
      maximumVersion?: string;
      retryable?: boolean;
    },
  ) {
    super(message);
    this.name = "RepositoryBrowserError";
    this.kind = kind;
    this.minimumVersion = options?.minimumVersion;
    this.maximumVersion = options?.maximumVersion;
    this.retryable = options?.retryable ?? false;
  }
}

export type RepositoryRoute =
  | { kind: "list" }
  | { kind: "detail"; routeId: string }
  | {
      kind: "blob";
      routeId: string;
      path: string;
      ref?: string;
      download: boolean;
    };

export function repositoryDetailPath(routeId: string): string {
  return `/repos/${encodeURIComponent(validRouteId(routeId))}`;
}

export function repositoryBlobPath(routeId: string, path: string): string {
  const encodedPath = validBlobPath(path)
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  return `${repositoryDetailPath(routeId)}/blob/${encodedPath}`;
}

export function repositoryDownloadPath(
  routeId: string,
  path: string,
  ref?: string,
): string {
  const query = new URLSearchParams({ download: "1" });
  if (ref !== undefined) query.set("ref", validGitRef(ref));
  return `${repositoryBlobPath(routeId, path)}?${query.toString()}`;
}

export function parseRepositoryRoute(value: string): RepositoryRoute {
  let url: URL;
  try {
    url = new URL(value, "https://routes.invalid");
  } catch {
    throw invalidRequest();
  }
  const segments = url.pathname.split("/");
  if (segments[0] !== "" || segments[1] !== "repos") throw invalidRequest();
  if (segments.length === 2 || (segments.length === 3 && segments[2] === "")) {
    return { kind: "list" };
  }
  const routeId = decodeSegment(segments[2]);
  if (segments.length === 3) return { kind: "detail", routeId };
  if (segments[3] !== "blob" || segments.length < 5) throw invalidRequest();
  const path = segments.slice(4).map(decodeSegment).join("/");
  const ref = url.searchParams.get("ref") ?? undefined;
  return {
    kind: "blob",
    routeId,
    path: validBlobPath(path),
    ref: ref === undefined ? undefined : validGitRef(ref),
    download: url.searchParams.get("download") === "1",
  };
}

export function repositoryResourcePath(repositoryId: string): string {
  return `${REPOSITORIES_API_PATH}/${validRepositoryId(repositoryId)}`;
}

export function repositoryTreeApiUrl(
  baseUrl: URL,
  repositoryId: string,
  ref: string,
  path?: string,
): URL {
  const url = new URL(`${repositoryResourcePath(repositoryId)}/tree`, baseUrl);
  url.searchParams.set("ref", validGitRef(ref));
  if (path !== undefined && path !== "") {
    url.searchParams.set("path", validBlobPath(path));
  }
  return url;
}

export function repositoryBlobApiUrl(
  baseUrl: URL,
  repositoryId: string,
  ref: string,
  path: string,
): URL {
  const url = new URL(`${repositoryResourcePath(repositoryId)}/blob`, baseUrl);
  url.searchParams.set("ref", validGitRef(ref));
  url.searchParams.set("path", validBlobPath(path));
  return url;
}

export function validRouteId(value: string): string {
  if (
    value.length === 0 ||
    value.length > 64 ||
    value.startsWith(".") ||
    value.includes("..") ||
    !/^[A-Za-z0-9._-]+$/.test(value)
  ) {
    throw invalidRequest();
  }
  return value;
}

export function validRepositoryId(value: string): string {
  if (
    !/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value) ||
    value === "00000000-0000-0000-0000-000000000000"
  ) {
    throw invalidRequest();
  }
  return value.toLowerCase();
}

export function validBlobPath(value: string): string {
  if (
    value.length === 0 ||
    new TextEncoder().encode(value).length > 4_096 ||
    value.startsWith("/") ||
    value.endsWith("/") ||
    containsControlCharacter(value)
  ) {
    throw invalidRequest();
  }
  const segments = value.split("/");
  if (
    segments.some(
      (segment) => segment === "" || segment === "." || segment === "..",
    )
  ) {
    throw invalidRequest();
  }
  return value;
}

export function validGitRef(value: string): string {
  if (
    value.length === 0 ||
    value.length > 1_024 ||
    value.startsWith("/") ||
    value.endsWith("/") ||
    value.startsWith(".") ||
    value.endsWith(".") ||
    value.includes("..") ||
    value.includes("@{") ||
    /[\s~^:?*[\\]/u.test(value) ||
    containsControlCharacter(value)
  ) {
    throw invalidRequest();
  }
  return value;
}

function decodeSegment(value: string | undefined): string {
  if (value === undefined) throw invalidRequest();
  try {
    return decodeURIComponent(value);
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

function invalidRequest(): RepositoryBrowserError {
  return new RepositoryBrowserError(
    "invalid_request",
    "The repository URL or request is invalid.",
  );
}
